# Google GenAI SDK Analysis for TokenLedger

> **SDK Version**: Latest (code-generated from Google API)
> **Analysis Date**: 2026-01-22
> **Current Coverage**: 0% (NEW PROVIDER - not yet supported)

## Overview

The Google GenAI SDK provides access to Gemini models for text generation, embeddings, image generation, and more. TokenLedger does not currently support Google as a provider. This document outlines the implementation requirements.

## SDK Structure

```
google/genai/
├── client.py               # Main Client class
├── _api_client.py          # BaseApiClient (HTTP layer)
├── models.py               # Models resource
├── caches.py               # Caching resource
├── batches.py              # Batch processing
├── files.py                # File management
├── live.py                 # Real-time bidirectional API
├── tunings.py              # Fine-tuning
└── types.py                # Response/request types
```

## Client Instantiation

```python
from google.genai import Client

# Gemini Developer API (API Key)
client = Client(api_key='your-api-key')

# Vertex AI (Service Account)
client = Client(
    vertexai=True,
    project='my-project-id',
    location='us-central1'
)

# Async access
async_client = client.aio
```

## API Methods with Cost Implications

### Primary Methods (High Priority)

| Resource | Method | Sync | Async | Endpoint | Cost Model |
|----------|--------|------|-------|----------|------------|
| **models** | `generate_content()` | ✅ | ✅ | `{model}:generateContent` | Input + Output tokens |
| **models** | `generate_content_stream()` | ✅ | ✅ | `{model}:streamGenerateContent` | Input + Output tokens |
| **models** | `embed_content()` | ✅ | ✅ | `{model}:batchEmbedContents` | Per embedding |
| **models** | `generate_images()` | ✅ | ✅ | `{model}:predict` | Per image |
| **models** | `edit_image()` | ✅ | ✅ | `{model}:predict` | Per image |
| **models** | `generate_videos()` | ✅ | ✅ | `{model}:predict` | Per video |

### Secondary Methods (Medium Priority)

| Resource | Method | Sync | Async | Endpoint | Cost Model |
|----------|--------|------|-------|----------|------------|
| **models** | `count_tokens()` | ✅ | ✅ | `{model}:countTokens` | Free |
| **models** | `compute_tokens()` | ✅ | ✅ | `{model}:computeTokens` | Free (Vertex only) |
| **caches** | `create()` | ✅ | ✅ | `cachedContents` | Cached token storage |
| **batches** | `create()` | ✅ | ✅ | `batchPredictionJobs` | Discounted batch rate |
| **live** | `connect()` | ❌ | ✅ | WebSocket | Real-time tokens |
| **tunings** | `create()` | ✅ | ✅ | `tunedModels` | Training cost |

### Non-Cost Methods

| Resource | Method | Notes |
|----------|--------|-------|
| files | `upload()`, `delete()` | Storage management |
| models | `get()`, `list()` | Metadata only |
| caches | `get()`, `delete()`, `update()` | Cache management |
| operations | `get()`, `list()` | Async job polling |

## Token Extraction

### Generate Content Response

```python
response.usage_metadata:
    .total_token_count          # Total tokens
    .prompt_token_count         # Input tokens
    .candidates_token_count     # Output tokens
    .cached_content_token_count # Cached tokens (discounted)
    .thoughts_token_count       # Extended thinking tokens
    .tool_use_prompt_token_count # Tool use tokens

    # Modality breakdown (for multimodal)
    .prompt_tokens_details: list[ModalityTokenCount]
        .modality               # "TEXT", "IMAGE", "VIDEO", "AUDIO"
        .token_count            # Tokens for this modality
```

### Streaming Response

```python
# Each chunk has partial usage_metadata
for chunk in response:
    chunk.usage_metadata.prompt_token_count      # Cumulative input
    chunk.usage_metadata.candidates_token_count  # Cumulative output
```

### Count Tokens Response

```python
response.total_tokens               # Estimated tokens
response.cached_content_token_count # Cached portion
```

## Recommended Implementation Approach

### Option 1: Method-Level Patching (Recommended)

```python
def patch_google():
    from google.genai import models

    # Sync methods
    _original_methods["generate_content"] = models.Models.generate_content
    models.Models.generate_content = _wrap_generate_content(models.Models.generate_content)

    _original_methods["embed_content"] = models.Models.embed_content
    models.Models.embed_content = _wrap_embed_content(models.Models.embed_content)

    # Async methods (via client.aio)
    _original_methods["async_generate_content"] = models.AsyncModels.generate_content
    models.AsyncModels.generate_content = _wrap_async_generate_content(...)
```

### Option 2: HTTP-Level Patching

```python
# Single interception point for all requests
from google.genai._api_client import BaseApiClient

_original_request = BaseApiClient.request
BaseApiClient.request = _wrap_request(_original_request)

_original_async_request = BaseApiClient.async_request
BaseApiClient.async_request = _wrap_async_request(_original_async_request)
```

**Advantage**: Captures all API calls with single patch
**Disadvantage**: Must parse request/response to determine operation type

## Pricing Data

```python
# Add to tokenledger/pricing.py

GOOGLE_PRICING = {
    # Gemini text models (per 1M tokens)
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash-lite": {"input": 0.02, "output": 0.10},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},

    # Cached content (typically 75% discount)
    "cached_multiplier": 0.25,

    # Embeddings
    "text-embedding-004": {"input": 0.00001},  # per token

    # Image generation (per image)
    "imagen-3.0-generate-002": 0.03,
    "imagen-3.0-fast-generate-001": 0.02,
}
```

## pydantic-ai Integration

From `/tmp/pydantic-ai/pydantic_ai_slim/pydantic_ai/models/google.py`:

```python
# Line ~1156 - How pydantic-ai extracts usage
def _metadata_as_usage(response, provider, provider_url):
    metadata = response.usage_metadata
    # Extracts:
    # - cached_content_tokens
    # - thoughts_tokens
    # - tool_use_prompt_tokens
    # - text_prompt_tokens, image_prompt_tokens (from modality details)
```

**Methods pydantic-ai calls**:
- `client.aio.models.generate_content()` - main generation
- `client.aio.models.generate_content_stream()` - streaming
- `client.aio.models.count_tokens()` - token estimation

## Database Schema Additions

For `token_ledger_events` table:

```sql
-- New fields for Google provider
ALTER TABLE token_ledger_events ADD COLUMN IF NOT EXISTS
    cached_content_tokens INTEGER DEFAULT 0;

ALTER TABLE token_ledger_events ADD COLUMN IF NOT EXISTS
    thoughts_tokens INTEGER DEFAULT 0;

-- Or use metadata_extra for Google-specific fields
```

## Event Structure

```json
{
  "provider": "google",
  "model": "gemini-2.0-flash",
  "input_tokens": 150,
  "output_tokens": 250,
  "cached_tokens": 50,
  "total_tokens": 400,
  "cost_usd": 0.00015,
  "status": "success",
  "metadata_extra": {
    "thoughts_tokens": 30,
    "tool_use_tokens": 10,
    "finish_reason": "STOP",
    "traffic_type": "UNCACHED"
  }
}
```

## Implementation Checklist

- [ ] Create `tokenledger/interceptors/google.py`
- [ ] Add `patch_google()` function
- [ ] Add `unpatch_google()` function
- [ ] Add Google pricing to `pricing.py`
- [ ] Add `calculate_cost()` support for Google models
- [ ] Export `patch_google` from `__init__.py`
- [ ] Add tests for Google interceptor
- [ ] Document Google support in quickstart

## Models to Support

| Model | Type | Pricing Tier |
|-------|------|--------------|
| gemini-2.0-flash | Text/Multimodal | Standard |
| gemini-2.0-flash-lite | Text | Budget |
| gemini-2.5-pro | Text/Multimodal | Premium |
| gemini-2.5-flash | Text/Multimodal | Standard |
| gemini-3-pro-preview | Text/Multimodal | Preview |
| text-embedding-004 | Embeddings | Standard |
| imagen-3.0-generate-002 | Image Gen | Per-image |
| imagen-3.0-fast-generate-001 | Image Gen | Per-image |

## References

- Google GenAI SDK: `/tmp/google-genai-sdk/google/genai/`
- pydantic-ai Google model: `/tmp/pydantic-ai/pydantic_ai_slim/pydantic_ai/models/google.py`
- Google AI pricing: https://ai.google.dev/pricing
