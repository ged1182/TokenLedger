# TokenLedger SDK Coverage Implementation Plan

> **Goal**: 100% coverage for cost-bearing APIs across OpenAI, Anthropic, and Google
> **Priority**: pydantic-ai compatibility first, then high-value APIs, then completeness

## Phase Overview

| Phase | Focus | Methods | Effort | Outcome |
|-------|-------|---------|--------|---------|
| **1** | pydantic-ai compatibility | 8 | Medium | pydantic-ai works with all providers |
| **2** | OpenAI high-value APIs | 12 | Medium | Audio, images, video tracked |
| **3** | Google provider | 10 | High | New provider fully supported |
| **4** | Completeness | 10+ | Low | 100% coverage |

---

## Phase 1: pydantic-ai Compatibility (CRITICAL)

**Goal**: Make TokenLedger work with pydantic-ai for all providers

### Step 1.1: Anthropic Beta Messages

**File**: `tokenledger/interceptors/anthropic.py`

| Task | Method | Class Path |
|------|--------|------------|
| 1.1.1 | Patch `create()` sync | `anthropic.resources.beta.messages.Messages` |
| 1.1.2 | Patch `create()` async | `anthropic.resources.beta.messages.AsyncMessages` |
| 1.1.3 | Patch `stream()` sync | `anthropic.resources.beta.messages.Messages` |
| 1.1.4 | Patch `stream()` async | `anthropic.resources.beta.messages.AsyncMessages` |

**Implementation**:
```python
# Add to patch_anthropic() function:
from anthropic.resources.beta import messages as beta_messages

# Reuse existing wrappers - same response structure
_original_methods["beta_messages_create"] = beta_messages.Messages.create
beta_messages.Messages.create = _wrap_messages_create(beta_messages.Messages.create)

_original_methods["beta_async_messages_create"] = beta_messages.AsyncMessages.create
beta_messages.AsyncMessages.create = _wrap_async_messages_create(beta_messages.AsyncMessages.create)

# Beta streaming
if hasattr(beta_messages.Messages, "stream"):
    _original_methods["beta_messages_stream"] = beta_messages.Messages.stream
    beta_messages.Messages.stream = _wrap_streaming_messages(beta_messages.Messages.stream)
```

**Tests needed**:
- `tests/test_interceptors/test_anthropic_beta.py`

### Step 1.2: OpenAI Responses API

**File**: `tokenledger/interceptors/openai.py`

| Task | Method | Class Path |
|------|--------|------------|
| 1.2.1 | Patch `create()` sync | `openai.resources.responses.Responses` |
| 1.2.2 | Patch `create()` async | `openai.resources.responses.AsyncResponses` |

**Implementation**:
```python
# Add wrapper function
def _wrap_responses_create(original_method):
    @functools.wraps(original_method)
    def wrapper(*args, **kwargs):
        # Similar to chat completions but different response structure
        # responses API uses: response.usage.input_tokens, output_tokens
        ...
    return wrapper

# Add to patch_openai():
from openai.resources.responses import responses

_original_methods["responses_create"] = responses.Responses.create
responses.Responses.create = _wrap_responses_create(responses.Responses.create)

_original_methods["async_responses_create"] = responses.AsyncResponses.create
responses.AsyncResponses.create = _wrap_async_responses_create(responses.AsyncResponses.create)
```

**Tests needed**:
- `tests/test_interceptors/test_openai_responses.py`

### Step 1.3: Google Provider (Basic)

**Files**:
- `tokenledger/interceptors/google.py` (NEW)
- `tokenledger/pricing.py` (update)
- `tokenledger/__init__.py` (update exports)

| Task | Method | Class Path |
|------|--------|------------|
| 1.3.1 | Create `patch_google()` | New file |
| 1.3.2 | Patch `generate_content()` sync | `google.genai.models.Models` |
| 1.3.3 | Patch `generate_content()` async | `google.genai.models.AsyncModels` |
| 1.3.4 | Add Google pricing | `tokenledger/pricing.py` |

**Implementation**:
```python
# tokenledger/interceptors/google.py
def _extract_tokens_from_response(response):
    """Extract tokens from Google GenAI response."""
    usage = getattr(response, "usage_metadata", None)
    if usage:
        return {
            "input_tokens": getattr(usage, "prompt_token_count", 0) or 0,
            "output_tokens": getattr(usage, "candidates_token_count", 0) or 0,
            "cached_tokens": getattr(usage, "cached_content_token_count", 0) or 0,
        }
    return {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}

def _wrap_generate_content(original_method):
    @functools.wraps(original_method)
    def wrapper(self, *, model, contents, **kwargs):
        # Track the call
        ...
    return wrapper

def patch_google():
    from google.genai import models

    _original_methods["generate_content"] = models.Models.generate_content
    models.Models.generate_content = _wrap_generate_content(models.Models.generate_content)

    _original_methods["async_generate_content"] = models.AsyncModels.generate_content
    models.AsyncModels.generate_content = _wrap_async_generate_content(...)
```

**Pricing to add**:
```python
GOOGLE_PRICING = {
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash-lite": {"input": 0.02, "output": 0.10},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    "text-embedding-004": {"input": 0.00001, "output": 0},
}
```

**Tests needed**:
- `tests/test_interceptors/test_google.py`

### Phase 1 Verification

```bash
# Test pydantic-ai compatibility
python -c "
import tokenledger
tokenledger.configure(database_url='postgresql://...')
tokenledger.patch_openai()
tokenledger.patch_anthropic()
tokenledger.patch_google()

from pydantic_ai import Agent

# These should all be tracked now
agent_openai = Agent(model='openai:gpt-4o')
agent_anthropic = Agent(model='anthropic:claude-3-5-sonnet-20241022')
agent_google = Agent(model='google-gla:gemini-2.0-flash')
"
```

---

## Phase 2: OpenAI High-Value APIs

**Goal**: Track audio, images, and video which represent ~25% of spend

### Step 2.1: Audio APIs

**File**: `tokenledger/interceptors/openai.py`

| Task | Method | Billing | Notes |
|------|--------|---------|-------|
| 2.1.1 | `audio.transcriptions.create()` sync | Per-minute | Track duration |
| 2.1.2 | `audio.transcriptions.create()` async | Per-minute | |
| 2.1.3 | `audio.translations.create()` sync | Per-minute | |
| 2.1.4 | `audio.translations.create()` async | Per-minute | |
| 2.1.5 | `audio.speech.create()` sync | Per-character | Track input length |
| 2.1.6 | `audio.speech.create()` async | Per-character | |

**New wrapper needed**:
```python
def _wrap_audio_transcription(original_method):
    """Track audio transcription calls - billed per minute."""
    @functools.wraps(original_method)
    def wrapper(*args, **kwargs):
        tracker = get_tracker()
        start_time = time.perf_counter()

        model = kwargs.get("model", "whisper-1")

        event = LLMEvent.fast_construct(
            provider="openai",
            model=model,
            request_type="transcription",
            endpoint="/v1/audio/transcriptions",
        )
        _apply_attribution_context(event)

        try:
            response = original_method(*args, **kwargs)
            event.duration_ms = (time.perf_counter() - start_time) * 1000
            event.status = "success"

            # Audio billing is per-minute, not per-token
            # Store duration in metadata or a new field
            # Cost calculation based on audio duration (from file)

            tracker.track(event)
            return response
        except Exception as e:
            # error handling
            ...
    return wrapper
```

**Pricing to add**:
```python
OPENAI_AUDIO_PRICING = {
    "whisper-1": 0.006,  # per minute
    "gpt-4o-transcribe": 0.006,
    "gpt-4o-mini-transcribe": 0.003,
    "tts-1": 0.015,  # per 1K characters
    "tts-1-hd": 0.030,
    "gpt-4o-mini-tts": 0.010,
}
```

### Step 2.2: Image APIs

**File**: `tokenledger/interceptors/openai.py`

| Task | Method | Billing | Notes |
|------|--------|---------|-------|
| 2.2.1 | `images.generate()` sync | Per-image | Track n, size, model |
| 2.2.2 | `images.generate()` async | Per-image | |
| 2.2.3 | `images.edit()` sync | Per-image | |
| 2.2.4 | `images.edit()` async | Per-image | |
| 2.2.5 | `images.create_variation()` sync | Per-image | |
| 2.2.6 | `images.create_variation()` async | Per-image | |

**Pricing to add**:
```python
OPENAI_IMAGE_PRICING = {
    "dall-e-3": {
        "1024x1024": 0.040,
        "1024x1792": 0.080,
        "1792x1024": 0.080,
    },
    "dall-e-3-hd": {
        "1024x1024": 0.080,
        "1024x1792": 0.120,
        "1792x1024": 0.120,
    },
    "dall-e-2": {
        "1024x1024": 0.020,
        "512x512": 0.018,
        "256x256": 0.016,
    },
}
```

### Step 2.3: Video APIs (Sora)

| Task | Method | Billing |
|------|--------|---------|
| 2.3.1 | `videos.create()` sync | Per-video |
| 2.3.2 | `videos.create()` async | Per-video |

---

## Phase 3: Google Provider Complete

**Goal**: Full Google GenAI coverage

### Step 3.1: Streaming

| Task | Method |
|------|--------|
| 3.1.1 | `generate_content_stream()` sync |
| 3.1.2 | `generate_content_stream()` async |

### Step 3.2: Embeddings

| Task | Method |
|------|--------|
| 3.2.1 | `embed_content()` sync |
| 3.2.2 | `embed_content()` async |

### Step 3.3: Images

| Task | Method |
|------|--------|
| 3.3.1 | `generate_images()` sync |
| 3.3.2 | `generate_images()` async |

### Step 3.4: Token Counting (Free, for auditing)

| Task | Method |
|------|--------|
| 3.4.1 | `count_tokens()` sync |
| 3.4.2 | `count_tokens()` async |

---

## Phase 4: Completeness

**Goal**: 100% coverage of all cost-bearing methods

### Step 4.1: OpenAI Remaining

| Task | Method | Priority |
|------|--------|----------|
| 4.1.1 | `completions.create()` | Low (legacy) |
| 4.1.2 | `moderations.create()` | Low (free) |
| 4.1.3 | `batches.create()` | Low |
| 4.1.4 | `fine_tuning.jobs.create()` | Low |

### Step 4.2: Anthropic Remaining

| Task | Method | Priority |
|------|--------|----------|
| 4.2.1 | `messages.count_tokens()` | Low (free) |
| 4.2.2 | `beta.messages.count_tokens()` | Low (free) |
| 4.2.3 | `messages.batches.create()` | Low |
| 4.2.4 | `beta.messages.batches.create()` | Low |
| 4.2.5 | Async streaming for standard messages | Medium |

### Step 4.3: Google Remaining

| Task | Method | Priority |
|------|--------|----------|
| 4.3.1 | `batches.create()` | Low |
| 4.3.2 | `caches.create()` | Low |
| 4.3.3 | `generate_videos()` | Low |

---

## File Changes Summary

### New Files
- `tokenledger/interceptors/google.py`
- `tests/test_interceptors/test_google.py`
- `tests/test_interceptors/test_anthropic_beta.py`
- `tests/test_interceptors/test_openai_responses.py`
- `tests/test_interceptors/test_openai_audio.py`
- `tests/test_interceptors/test_openai_images.py`

### Modified Files
- `tokenledger/interceptors/openai.py` - Add responses, audio, images, video
- `tokenledger/interceptors/anthropic.py` - Add beta.messages
- `tokenledger/pricing.py` - Add Google, audio, image pricing
- `tokenledger/__init__.py` - Export `patch_google`, `unpatch_google`
- `pyproject.toml` - Add `google-genai` as optional dependency

---

## Testing Strategy

### Unit Tests (for each new method)
```python
def test_anthropic_beta_messages_create():
    """Test that beta.messages.create is tracked."""
    with patch("anthropic.resources.beta.messages.Messages.create") as mock:
        mock.return_value = MockResponse(usage=MockUsage(input_tokens=100, output_tokens=50))
        # ... verify event tracked correctly
```

### Integration Tests
```python
@requires_openai_key
def test_pydantic_ai_openai():
    """Test pydantic-ai with OpenAI is tracked."""
    patch_openai()
    agent = Agent(model="openai:gpt-4o")
    result = agent.run_sync("Hello")
    # Verify event in database

@requires_anthropic_key
def test_pydantic_ai_anthropic():
    """Test pydantic-ai with Anthropic is tracked."""
    patch_anthropic()
    agent = Agent(model="anthropic:claude-3-5-sonnet-20241022")
    result = agent.run_sync("Hello")
    # Verify event in database
```

---

## Success Metrics

| Phase | OpenAI | Anthropic | Google | pydantic-ai |
|-------|--------|-----------|--------|-------------|
| Current | 8% | 13% | 0% | ❌ |
| Phase 1 | 15% | 50% | 30% | ✅ |
| Phase 2 | 70% | 50% | 30% | ✅ |
| Phase 3 | 70% | 50% | 80% | ✅ |
| Phase 4 | 100% | 100% | 100% | ✅ |

---

## Execution Order

```
Phase 1 (pydantic-ai - CRITICAL)
├── 1.1 Anthropic beta.messages (4 methods)
├── 1.2 OpenAI responses (2 methods)
└── 1.3 Google basic (2 methods + pricing)

Phase 2 (OpenAI high-value)
├── 2.1 Audio APIs (6 methods)
├── 2.2 Image APIs (6 methods)
└── 2.3 Video APIs (2 methods)

Phase 3 (Google complete)
├── 3.1 Streaming (2 methods)
├── 3.2 Embeddings (2 methods)
├── 3.3 Images (2 methods)
└── 3.4 Token counting (2 methods)

Phase 4 (Completeness)
├── 4.1 OpenAI remaining
├── 4.2 Anthropic remaining
└── 4.3 Google remaining
```

---

## Dependencies to Add

```toml
# pyproject.toml
[project.optional-dependencies]
google = ["google-genai>=1.0.0"]
all = ["openai>=1.0.0", "anthropic>=0.18.0", "google-genai>=1.0.0"]
```
