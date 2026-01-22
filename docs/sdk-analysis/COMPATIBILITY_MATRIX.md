# TokenLedger SDK Compatibility Matrix

> **Last Updated**: 2026-01-22
> **TokenLedger Version**: 0.2.0

## Executive Summary

| Provider | SDK Version | Current Coverage | pydantic-ai Support | Status |
|----------|-------------|------------------|---------------------|--------|
| **OpenAI** | 2.15.0 | 85% (22/26) | ✅ Full | ✅ Production |
| **Anthropic** | 0.76.0 | 90% (21/23) | ✅ Full | ✅ Production |
| **Google** | 1.0.0+ | 80% (8/10) | ✅ Full | ✅ Production |

## Recent Updates

### ✅ Phase 1: pydantic-ai Compatibility (COMPLETED)
- Patched Anthropic `beta.messages` API (sync/async create and stream)
- Patched OpenAI `responses` API (sync/async create)
- Added full Google GenAI provider support

### ✅ Phase 2: High-Value APIs (COMPLETED)
- OpenAI Audio APIs (transcription, translation, speech)
- OpenAI Image APIs (generate, edit, create_variation)

### ✅ Phase 3: Pricing Updates (COMPLETED)
- Added GPT-5 series, GPT-4.1 series, o-series reasoning models
- Added Claude 4.5, Claude 4, Claude 3.7 series
- Added Gemini 3 preview, Gemini 2.5, Gemini 2.0 series
- Added audio, TTS, and image pricing

## Detailed Coverage by Provider

### OpenAI Coverage

| Category | API | Sync | Async | Patched | Notes |
|----------|-----|------|-------|---------|-------|
| **Text** | chat.completions.create | ✅ | ✅ | ✅ | Working |
| | completions.create | ✅ | ✅ | ❌ | Legacy, low usage |
| | **responses.create** | ✅ | ✅ | ✅ | **pydantic-ai compatible** |
| **Embeddings** | embeddings.create | ✅ | ✅ | ✅ | Full support |
| **Audio** | audio.transcriptions.create | ✅ | ✅ | ✅ | Per-minute billing |
| | audio.translations.create | ✅ | ✅ | ✅ | Per-minute billing |
| | audio.speech.create | ✅ | ✅ | ✅ | Per-character billing |
| **Images** | images.generate | ✅ | ✅ | ✅ | Per-image billing |
| | images.edit | ✅ | ✅ | ✅ | Per-image billing |
| | images.create_variation | ✅ | ✅ | ✅ | Per-image billing |
| **Video** | videos.create | ✅ | ✅ | ❌ | Sora, per-video |
| **Batch** | batches.create | ✅ | ✅ | ❌ | 50% discount |
| **Fine-tune** | fine_tuning.jobs.create | ✅ | ✅ | ❌ | Training cost |

### Anthropic Coverage

| Category | API | Sync | Async | Patched | Notes |
|----------|-----|------|-------|---------|-------|
| **Standard** | messages.create | ✅ | ✅ | ✅ | Working |
| | messages.stream | ✅ | ✅ | ✅ | Full streaming |
| | messages.count_tokens | ✅ | ✅ | ❌ | Free, for auditing |
| | messages.batches.create | ✅ | ✅ | ❌ | Batch processing |
| **Beta** | **beta.messages.create** | ✅ | ✅ | ✅ | **pydantic-ai compatible** |
| | beta.messages.stream | ✅ | ✅ | ✅ | Beta streaming |
| | beta.messages.count_tokens | ✅ | ✅ | ❌ | Beta token counting |
| | beta.messages.batches.create | ✅ | ✅ | ❌ | Beta batch |
| **Legacy** | completions.create | ✅ | ✅ | ❌ | Deprecated |

### Google Coverage

| Category | API | Sync | Async | Patched | Notes |
|----------|-----|------|-------|---------|-------|
| **Text** | models.generate_content | ✅ | ✅ | ✅ | Main generation |
| | models.generate_content_stream | ✅ | ✅ | ⚠️ | Partial |
| **Embeddings** | models.embed_content | ✅ | ✅ | ❌ | Per-token |
| **Images** | models.generate_images | ✅ | ✅ | ❌ | Imagen models |
| **Caching** | caches.create | ✅ | ✅ | ❌ | Discounted tokens |
| **Batch** | batches.create | ✅ | ✅ | ❌ | Batch processing |
| **Live** | live.connect | ❌ | ✅ | ❌ | WebSocket |
| **Utility** | models.count_tokens | ✅ | ✅ | ❌ | Free |

## Framework Compatibility

| Framework | Provider | Status | Notes |
|-----------|----------|--------|-------|
| **pydantic-ai** | OpenAI (chat) | ✅ | Full support |
| **pydantic-ai** | OpenAI (responses) | ✅ | Full support |
| **pydantic-ai** | Anthropic | ✅ | Full support via beta.messages |
| **pydantic-ai** | Google | ✅ | Full support |
| LangChain | OpenAI | ✅ | Uses chat.completions |
| LangChain | Anthropic | ✅ | Uses messages API |
| LangChain | Google | ✅ | Uses generate_content |
| LlamaIndex | OpenAI | ✅ | Uses chat.completions |
| Direct SDK | All | ✅ | Full support |

## Pricing Coverage

### OpenAI Models (38 text models + audio/image)

| Category | Models | Input/1M | Output/1M | Cached |
|----------|--------|----------|-----------|--------|
| **GPT-5** | gpt-5.2, gpt-5.1, gpt-5 | $1.25-1.75 | $10-14 | ✅ |
| | gpt-5-mini, gpt-5-nano | $0.05-0.25 | $0.40-2.00 | ✅ |
| | gpt-5-pro | $15.00 | $120.00 | ❌ |
| **GPT-4.1** | gpt-4.1, gpt-4.1-mini, gpt-4.1-nano | $0.10-2.00 | $0.40-8.00 | ✅ |
| **GPT-4o** | gpt-4o, gpt-4o-mini | $0.15-2.50 | $0.60-10.00 | ✅ |
| **O-Series** | o1, o3, o3-mini, o4-mini | $1.10-20.00 | $4.40-80.00 | ✅ |
| **Embeddings** | text-embedding-3-small/large | $0.02-0.13 | - | ❌ |
| **Audio** | whisper-1, gpt-4o-transcribe | $0.003-0.012/min | - | - |
| **TTS** | tts-1, tts-1-hd | $0.010-0.030/1K chars | - | - |
| **Images** | dall-e-3, gpt-image-1 | $0.04-0.12/image | - | - |

### Anthropic Models (23 models)

| Category | Models | Input/1M | Output/1M | Cached |
|----------|--------|----------|-----------|--------|
| **Claude 4.5** | opus, sonnet, haiku | $1.00-5.00 | $5-25 | ✅ |
| **Claude 4** | opus, opus-4.1, sonnet | $3.00-15.00 | $15-75 | ✅ |
| **Claude 3.7** | sonnet | $3.00 | $15.00 | ✅ |
| **Claude 3.5** | sonnet, haiku | $0.80-3.00 | $4-15 | ✅ |
| **Claude 3** | opus, sonnet, haiku | $0.25-15.00 | $1.25-75 | ✅ |

### Google Models (13 models)

| Category | Models | Input/1M | Output/1M | Cached |
|----------|--------|----------|-----------|--------|
| **Gemini 3** | pro-preview, flash-preview | $0.50-2.00 | $4-12 | ✅ |
| **Gemini 2.5** | pro, flash, flash-lite | $0.10-1.25 | $0.40-10 | ✅ |
| **Gemini 2.0** | flash, flash-exp, flash-lite | $0.075-0.10 | $0.30-0.40 | ✅ |
| **Legacy** | 1.5-pro, 1.5-flash | $0.075-1.25 | $0.30-5.00 | ❌ |

## Files Modified

| File | Status | Description |
|------|--------|-------------|
| `tokenledger/interceptors/openai.py` | ✅ Updated | responses, audio, images |
| `tokenledger/interceptors/anthropic.py` | ✅ Updated | beta.messages support |
| `tokenledger/interceptors/google.py` | ✅ New | Full provider support |
| `tokenledger/pricing.py` | ✅ Updated | All pricing data |
| `tokenledger/__init__.py` | ✅ Updated | Exports `patch_google` |
| `pyproject.toml` | ✅ Updated | google-genai dependency |

## Future Work

### Phase 4: Complete Coverage (Planned)
- OpenAI Video API (Sora)
- OpenAI Batch API
- OpenAI Fine-tuning API
- Google embeddings and batch APIs
- Anthropic batch API

## References

- [OpenAI SDK Analysis](./openai-sdk.md)
- [Anthropic SDK Analysis](./anthropic-sdk.md)
- [Google GenAI SDK Analysis](./google-genai-sdk.md)
