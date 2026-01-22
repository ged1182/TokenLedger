# OpenAI SDK Analysis for TokenLedger

> **SDK Version**: 2.15.0
> **Analysis Date**: 2026-01-22
> **Current Coverage**: ~10% (3 of 30+ cost-bearing methods)

## Overview

The OpenAI Python SDK provides access to multiple AI services including text generation, embeddings, audio, images, and video. TokenLedger currently patches only chat completions and embeddings, missing significant cost-bearing endpoints.

## SDK Structure

```
openai/resources/
├── chat/
│   └── completions.py        # Chat completions (PATCHED)
├── completions.py            # Legacy completions (NOT PATCHED)
├── embeddings.py             # Embeddings (PATCHED)
├── responses/                # New Responses API (NOT PATCHED)
│   └── responses.py
├── audio/
│   ├── transcriptions.py     # Whisper (NOT PATCHED)
│   ├── translations.py       # Audio translation (NOT PATCHED)
│   └── speech.py             # TTS (NOT PATCHED)
├── images.py                 # DALL-E (NOT PATCHED)
├── videos.py                 # Sora (NOT PATCHED)
├── moderations.py            # Content moderation (NOT PATCHED)
├── batches.py                # Batch API (NOT PATCHED)
├── fine_tuning/              # Fine-tuning (NOT PATCHED)
├── beta/
│   └── threads/runs/         # Assistants API (NOT PATCHED, deprecated)
└── realtime/                 # Real-time API (NOT PATCHED)
```

## Currently Patched Methods

| Resource | Method | Sync | Async | Module Path |
|----------|--------|------|-------|-------------|
| chat.completions | `create()` | ✅ | ✅ | `openai.resources.chat.completions.Completions` |
| embeddings | `create()` | ✅ | ❌ | `openai.resources.embeddings.Embeddings` |

## Gap Analysis: Missing Methods

### Tier 1 - High Priority (Token/Cost-Bearing)

| Resource | Method | Sync | Async | Endpoint | Cost Model | Priority |
|----------|--------|------|-------|----------|------------|----------|
| **responses** | `create()` | ✅ | ✅ | `/v1/responses` | Token-based | 🔴 CRITICAL (pydantic-ai uses this) |
| completions | `create()` | ✅ | ✅ | `/v1/completions` | Token-based | 🟠 HIGH |
| audio.transcriptions | `create()` | ✅ | ✅ | `/v1/audio/transcriptions` | Per-minute | 🟠 HIGH |
| audio.translations | `create()` | ✅ | ✅ | `/v1/audio/translations` | Per-minute | 🟠 HIGH |
| audio.speech | `create()` | ✅ | ✅ | `/v1/audio/speech` | Per-character | 🟠 HIGH |
| images | `generate()` | ✅ | ✅ | `/v1/images/generations` | Per-image | 🟠 HIGH |
| images | `edit()` | ✅ | ✅ | `/v1/images/edits` | Per-image | 🟠 HIGH |
| images | `create_variation()` | ✅ | ✅ | `/v1/images/variations` | Per-image | 🟠 HIGH |
| videos | `create()` | ✅ | ✅ | `/v1/videos/generations` | Per-video | 🟠 HIGH |
| videos | `remix()` | ✅ | ✅ | `/v1/videos/remix` | Per-video | 🟠 HIGH |

### Tier 2 - Medium Priority

| Resource | Method | Sync | Async | Endpoint | Cost Model | Priority |
|----------|--------|------|-------|----------|------------|----------|
| moderations | `create()` | ✅ | ✅ | `/v1/moderations` | Free (audit) | 🟡 MEDIUM |
| batches | `create()` | ✅ | ✅ | `/v1/batches` | 50% discount | 🟡 MEDIUM |
| fine_tuning.jobs | `create()` | ✅ | ✅ | `/v1/fine_tuning/jobs` | Training cost | 🟡 MEDIUM |
| beta.threads.runs | `create()` | ✅ | ✅ | `/v1/threads/{id}/runs` | Token-based | 🟡 MEDIUM (deprecated) |

### Tier 3 - Lower Priority

| Resource | Method | Sync | Async | Endpoint | Cost Model | Priority |
|----------|--------|------|-------|----------|------------|----------|
| realtime | `connect()` | ✅ | ✅ | WebSocket | Token-based | 🟢 LOW (complex) |
| embeddings | `create()` | ❌ | ✅ | `/v1/embeddings` | Token-based | 🟢 LOW (async missing) |

## Token Extraction by API Type

### Token-Based APIs
```python
# Chat completions, completions, responses
response.usage.prompt_tokens      # Input tokens
response.usage.completion_tokens  # Output tokens
response.usage.total_tokens       # Total
response.usage.prompt_tokens_details.cached_tokens  # Cached (if applicable)
```

### Non-Token APIs

| API | Cost Metric | How to Extract |
|-----|-------------|----------------|
| Audio Transcription | Duration (minutes) | Request file duration |
| Audio TTS | Characters | `len(input_text)` |
| Images | Count × Size | `n` parameter, `size` parameter |
| Videos | Duration × Resolution | `seconds`, `size` parameters |

## Pricing Data Requirements

Add to `tokenledger/pricing.py`:

```python
# Audio pricing (per minute)
OPENAI_AUDIO_PRICING = {
    "whisper-1": 0.006,
    "gpt-4o-transcribe": 0.006,
    "gpt-4o-mini-transcribe": 0.003,
}

# TTS pricing (per 1K characters)
OPENAI_TTS_PRICING = {
    "tts-1": 0.015,
    "tts-1-hd": 0.030,
    "gpt-4o-mini-tts": 0.010,
}

# Image pricing (per image)
OPENAI_IMAGE_PRICING = {
    "dall-e-3-1024x1024": 0.040,
    "dall-e-3-1024x1792": 0.080,
    "dall-e-3-1792x1024": 0.080,
    "dall-e-3-hd-1024x1024": 0.080,
    "dall-e-3-hd-1024x1792": 0.120,
    "dall-e-3-hd-1792x1024": 0.120,
    "dall-e-2-1024x1024": 0.020,
    "dall-e-2-512x512": 0.018,
    "dall-e-2-256x256": 0.016,
}

# Video pricing (per video, varies by duration/size)
OPENAI_VIDEO_PRICING = {
    "sora-2-480p": {"4s": 0.10, "8s": 0.20, "12s": 0.30},
    "sora-2-720p": {"4s": 0.20, "8s": 0.40, "12s": 0.60},
    "sora-2-1080p": {"4s": 0.50, "8s": 1.00, "12s": 1.50},
}
```

## Implementation Recommendations

### 1. Responses API (Critical for pydantic-ai)

```python
# In patch_openai():
from openai.resources.responses import responses

_original_methods["responses_create"] = responses.Responses.create
responses.Responses.create = _wrap_responses_create(responses.Responses.create)

_original_methods["async_responses_create"] = responses.AsyncResponses.create
responses.AsyncResponses.create = _wrap_async_responses_create(responses.AsyncResponses.create)
```

### 2. Suggested File Organization

```
tokenledger/interceptors/
├── openai/
│   ├── __init__.py       # Unified patch_openai()
│   ├── chat.py           # chat.completions, responses
│   ├── text.py           # completions, embeddings
│   ├── audio.py          # transcriptions, translations, speech
│   ├── images.py         # DALL-E
│   ├── video.py          # Sora
│   └── batch.py          # batches, fine-tuning
└── openai.py             # Legacy (current file, deprecated)
```

## Coverage Summary

| Category | Methods | Patched | Coverage |
|----------|---------|---------|----------|
| Text Generation | 4 | 1 | 25% |
| Embeddings | 2 | 1 | 50% |
| Audio | 6 | 0 | 0% |
| Images | 6 | 0 | 0% |
| Video | 4 | 0 | 0% |
| Batch/Fine-tune | 4 | 0 | 0% |
| **Total** | **26+** | **2** | **~8%** |

## References

- OpenAI SDK: `/tmp/openai-sdk/src/openai/`
- TokenLedger interceptor: `tokenledger/interceptors/openai.py`
- pydantic-ai OpenAI model: `/tmp/pydantic-ai/pydantic_ai_slim/pydantic_ai/models/openai.py`
