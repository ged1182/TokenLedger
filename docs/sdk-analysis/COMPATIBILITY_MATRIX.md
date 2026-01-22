# TokenLedger SDK Compatibility Matrix

> **Last Updated**: 2026-01-22
> **TokenLedger Version**: 0.1.0

## Executive Summary

| Provider | SDK Version | Current Coverage | pydantic-ai Support | Priority |
|----------|-------------|------------------|---------------------|----------|
| **OpenAI** | 2.15.0 | 8% (2/26) | ⚠️ Partial | 🔴 HIGH |
| **Anthropic** | 0.76.0 | 13% (3/23) | ❌ None | 🔴 CRITICAL |
| **Google** | Latest | 0% (0/12) | ❌ None | 🟠 NEW |

## Critical Issues

### 1. pydantic-ai Incompatibility (CRITICAL)

```
❌ pydantic-ai calls client.beta.messages.create() for Anthropic
❌ TokenLedger only patches client.messages.create()
❌ These are DIFFERENT classes - ZERO coverage for pydantic-ai + Anthropic

⚠️ pydantic-ai can use client.responses.create() for OpenAI (new API)
⚠️ TokenLedger doesn't patch responses API
```

### 2. Missing High-Value APIs

| API | Est. % of User Spend | Current Status |
|-----|---------------------|----------------|
| OpenAI Chat Completions | 40% | ✅ Tracked |
| **Anthropic Beta Messages** | **25%** | ❌ NOT Tracked |
| OpenAI Images (DALL-E) | 15% | ❌ NOT Tracked |
| OpenAI Audio (Whisper/TTS) | 10% | ❌ NOT Tracked |
| **OpenAI Responses API** | **5%** | ❌ NOT Tracked |
| Google Gemini | 5% | ❌ NOT Tracked |

## Detailed Coverage by Provider

### OpenAI Coverage

| Category | API | Sync | Async | Patched | Notes |
|----------|-----|------|-------|---------|-------|
| **Text** | chat.completions.create | ✅ | ✅ | ✅ | Working |
| | completions.create | ✅ | ✅ | ❌ | Legacy, still used |
| | **responses.create** | ✅ | ✅ | ❌ | **pydantic-ai uses this** |
| **Embeddings** | embeddings.create | ✅ | ❌ | ⚠️ | Async missing |
| **Audio** | audio.transcriptions.create | ✅ | ✅ | ❌ | Per-minute billing |
| | audio.translations.create | ✅ | ✅ | ❌ | Per-minute billing |
| | audio.speech.create | ✅ | ✅ | ❌ | Per-character billing |
| **Images** | images.generate | ✅ | ✅ | ❌ | Per-image billing |
| | images.edit | ✅ | ✅ | ❌ | Per-image billing |
| | images.create_variation | ✅ | ✅ | ❌ | Per-image billing |
| **Video** | videos.create | ✅ | ✅ | ❌ | Sora, per-video |
| **Batch** | batches.create | ✅ | ✅ | ❌ | 50% discount |
| **Fine-tune** | fine_tuning.jobs.create | ✅ | ✅ | ❌ | Training cost |

### Anthropic Coverage

| Category | API | Sync | Async | Patched | Notes |
|----------|-----|------|-------|---------|-------|
| **Standard** | messages.create | ✅ | ✅ | ✅ | Working |
| | messages.stream | ✅ | ❌ | ⚠️ | Async missing |
| | messages.count_tokens | ✅ | ✅ | ❌ | Free, for auditing |
| | messages.batches.create | ✅ | ✅ | ❌ | Batch processing |
| **Beta** | **beta.messages.create** | ✅ | ✅ | ❌ | **pydantic-ai uses this!** |
| | **beta.messages.parse** | ✅ | ✅ | ❌ | Structured output |
| | **beta.messages.stream** | ✅ | ✅ | ❌ | Beta streaming |
| | **beta.messages.tool_runner** | ✅ | ✅ | ❌ | Tool execution |
| | beta.messages.count_tokens | ✅ | ✅ | ❌ | Beta token counting |
| | beta.messages.batches.create | ✅ | ✅ | ❌ | Beta batch |
| **Legacy** | completions.create | ✅ | ✅ | ❌ | Deprecated |

### Google Coverage (NEW PROVIDER)

| Category | API | Sync | Async | Patched | Notes |
|----------|-----|------|-------|---------|-------|
| **Text** | models.generate_content | ✅ | ✅ | ❌ | Main generation |
| | models.generate_content_stream | ✅ | ✅ | ❌ | Streaming |
| **Embeddings** | models.embed_content | ✅ | ✅ | ❌ | Per-token |
| **Images** | models.generate_images | ✅ | ✅ | ❌ | Imagen models |
| | models.edit_image | ✅ | ✅ | ❌ | Image editing |
| **Video** | models.generate_videos | ✅ | ✅ | ❌ | Vids models |
| **Caching** | caches.create | ✅ | ✅ | ❌ | Discounted tokens |
| **Batch** | batches.create | ✅ | ✅ | ❌ | Batch processing |
| **Live** | live.connect | ❌ | ✅ | ❌ | WebSocket |
| **Utility** | models.count_tokens | ✅ | ✅ | ❌ | Free |

## Framework Compatibility

| Framework | Provider | Works? | Issue |
|-----------|----------|--------|-------|
| **pydantic-ai** | OpenAI (chat) | ✅ | None |
| **pydantic-ai** | OpenAI (responses) | ❌ | responses.create not patched |
| **pydantic-ai** | Anthropic | ❌ | **beta.messages not patched** |
| **pydantic-ai** | Google | ❌ | Provider not supported |
| LangChain | OpenAI | ✅ | Uses chat.completions |
| LangChain | Anthropic | ⚠️ | May use beta API |
| LlamaIndex | OpenAI | ✅ | Uses chat.completions |
| Direct SDK | All | ⚠️ | Depends on API used |

## Implementation Priority

### Phase 1: Critical (pydantic-ai compatibility)

1. **Anthropic beta.messages** - 6 methods
   - `beta.messages.Messages.create()` (sync)
   - `beta.messages.AsyncMessages.create()` (async)
   - `beta.messages.Messages.parse()` (sync)
   - `beta.messages.AsyncMessages.parse()` (async)
   - `beta.messages.Messages.stream()` (sync)
   - `beta.messages.AsyncMessages.stream()` (async)

2. **OpenAI responses** - 2 methods
   - `responses.Responses.create()` (sync)
   - `responses.AsyncResponses.create()` (async)

### Phase 2: High Value

3. **OpenAI Audio** - 6 methods
   - transcriptions (sync/async)
   - translations (sync/async)
   - speech (sync/async)

4. **OpenAI Images** - 6 methods
   - generate (sync/async)
   - edit (sync/async)
   - create_variation (sync/async)

### Phase 3: New Provider

5. **Google GenAI** - 8 methods
   - generate_content (sync/async)
   - generate_content_stream (sync/async)
   - embed_content (sync/async)
   - count_tokens (sync/async)

### Phase 4: Complete Coverage

6. **Remaining methods** - batch, fine-tuning, video, etc.

## Effort Estimates

| Phase | Methods | Complexity | Files to Change |
|-------|---------|------------|-----------------|
| Phase 1 | 8 | Medium | 2 (existing interceptors) |
| Phase 2 | 12 | Medium | 2 (existing interceptors) |
| Phase 3 | 8 | High | 3 (new interceptor + pricing + init) |
| Phase 4 | 15+ | Medium | 2-3 |

## Testing Requirements

For each new method:
- [ ] Unit test with mocked response
- [ ] Token extraction verification
- [ ] Cost calculation verification
- [ ] Error handling test
- [ ] Async variant test (if applicable)
- [ ] Streaming test (if applicable)

## Files to Modify

| File | Changes Needed |
|------|----------------|
| `tokenledger/interceptors/openai.py` | Add responses, audio, images patches |
| `tokenledger/interceptors/anthropic.py` | Add beta.messages patches |
| `tokenledger/interceptors/google.py` | **NEW FILE** |
| `tokenledger/pricing.py` | Add Google pricing, audio/image pricing |
| `tokenledger/__init__.py` | Export `patch_google` |
| `tests/test_interceptors/` | Tests for all new patches |

## Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| OpenAI coverage | 8% | 80% |
| Anthropic coverage | 13% | 90% |
| Google coverage | 0% | 80% |
| pydantic-ai compatible | ❌ | ✅ |
| All cost-bearing APIs tracked | ❌ | ✅ |

## References

- [OpenAI SDK Analysis](./openai-sdk.md)
- [Anthropic SDK Analysis](./anthropic-sdk.md)
- [Google GenAI SDK Analysis](./google-genai-sdk.md)
