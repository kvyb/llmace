# LLMACE Setup Guide

Quick guide to get LLMACE testing up and running.

## 🚀 Quick Setup (3 Steps)

### 1. Copy the environment template
```bash
cp env.example .env
```

### 2. Get your API keys

**Recommended Setup: OpenRouter + OpenAI**
- **OpenRouter** (for LLM): [openrouter.ai/keys](https://openrouter.ai/keys)
- **OpenAI** (for embeddings): [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

*Why both?* OpenRouter gives you access to 100+ models (Grok, GPT, Claude, etc.), while OpenAI provides the best embedding quality.

**Alternative:** Just OpenRouter or just OpenAI works too!

### 3. Edit `.env` and add your keys
```bash
nano .env  # or use your favorite editor
```

**Recommended (both keys):**
```bash
OPENROUTER_API_KEY=sk-or-v1-your-actual-key-here
OPENAI_API_KEY=sk-your-actual-key-here
```

**Alternative (OpenRouter only):**
```bash
OPENROUTER_API_KEY=sk-or-v1-your-actual-key-here
```

**Alternative (OpenAI only):**
```bash
OPENAI_API_KEY=sk-your-actual-key-here
```

### Priority Logic
The system automatically uses:
- **OpenRouter for LLM** (if `OPENROUTER_API_KEY` exists)
- **OpenAI for embeddings** (if `OPENAI_API_KEY` exists) ← Recommended for best quality
- Falls back intelligently if only one key is present

That's it! The defaults are already optimized.

## 🧪 Running Tests

### Quick Test (Verify Installation)
```bash
cd tests
python quick_test.py
```

Expected: `✅ ALL TESTS PASSED!`

### Effectiveness Test (Prove LLMACE Works)
```bash
cd tests
python test_llmace_effectiveness.py
```

This will:
1. Test with **Grok 2 Fast** (fast, good quality)
2. Test with **GPT-4.5 Turbo** (highest quality)
3. Use **Gemini 2.0 Flash** as judge (fast, accurate evaluation)
4. Show improvement metrics for each model

Expected: `✅ LLMACE shows consistent improvement across models!`

## 🎯 Default Configuration

The `env.example` comes pre-configured with optimized defaults:

### Test Models
- **Grok 2 Fast** (`x-ai/grok-2-fast`)
  - Fast inference for quick iteration
  - Good quality
  - Cost-effective

- **GPT-4.5 Turbo** (`openai/gpt-4.5-turbo`)
  - Highest quality
  - Proves LLMACE helps even top-tier models

### Judge Model
- **Gemini 2.0 Flash** (`google/gemini-2.0-flash-exp:free`)
  - Excellent evaluation capabilities
  - Fast response times
  - Cost-effective
  - Free tier available on OpenRouter

## 📊 Understanding Results

### Single Model Output
```
📊 BASELINE (No LLMACE):
  Average Score: 7.2/10
  Accuracy: 75.0%

🚀 WITH LLMACE:
  Average Score: 8.5/10
  Accuracy: 87.5%

📈 IMPROVEMENT:
  Score Delta: +1.3 points
  Percentage Improvement: 18.1%
```

### Multi-Model Output
```
MULTI-MODEL COMPARISON SUMMARY
📊 x-ai/grok-2-fast:     Improvement: 18.1% | Score: 8.5/10
📊 openai/gpt-4.5-turbo: Improvement: 14.8% | Score: 9.3/10
```

**What this means:**
- LLMACE improves performance across different model types
- Fast models (Grok) often see bigger improvements
- Even top-tier models (GPT-4.5) benefit from LLMACE

## ⚙️ Customization

### Use Different Models

Edit `.env`:
```bash
# Test with Claude instead
TEST_MODELS=anthropic/claude-3.5-sonnet,anthropic/claude-3-opus

# Or mix different providers
TEST_MODELS=x-ai/grok-2-fast,anthropic/claude-3.5-sonnet,openai/gpt-4.5-turbo
```

### Use Single Model
```bash
# Just test with one model
TEST_MODEL=x-ai/grok-2-fast

# Comment out or remove TEST_MODELS line
# TEST_MODELS=...
```

### Change Judge Model
```bash
# Use GPT-4 as judge
JUDGE_MODEL=openai/gpt-4

# Or Claude
JUDGE_MODEL=anthropic/claude-3.5-sonnet
```

## 💡 Tips

### Cost Optimization
- Use free tier models when available: `google/gemini-2.0-flash-exp:free`
- Start with faster models (Grok 2 Fast) for iteration
- Use premium models (GPT-4.5) for final validation

### Testing Strategy
1. **Development**: Use `TEST_MODEL=x-ai/grok-2-fast` (fast, cheap)
2. **Validation**: Use `TEST_MODELS=x-ai/grok-2-fast,openai/gpt-4.5-turbo` (comprehensive)
3. **Production**: Monitor with your actual production model

### Interpreting Results
- **Good**: 10-30% improvement
- **Excellent**: 30%+ improvement
- **Needs tuning**: <5% improvement (adjust prompts/config)

## 🔍 Troubleshooting

### "No API key found"
- Make sure you copied `env.example` to `.env`
- Check that your API key is correct
- Verify the key starts with `sk-or-v1-` for OpenRouter

### "Model not found"
- Check OpenRouter's model catalog: [openrouter.ai/models](https://openrouter.ai/models)
- Verify model name spelling (case-sensitive)
- Some models require credits on OpenRouter

### Tests taking too long
- Use faster models: `x-ai/grok-2-fast` or `google/gemini-2.0-flash-exp:free`
- Reduce number of test tasks (edit `create_test_tasks()` in test file)
- Test with single model instead of multiple

### No improvement detected
- Make sure `auto_update=True` in reflection calls
- Check that LLM client is properly configured
- Try with more test iterations (10+ tasks)

## 📚 Next Steps

After successful testing:

1. **Integrate into your app** - See [README.md](README.md) for usage examples
2. **Customize prompts** - Adjust reflection/curation prompts for your domain
3. **Monitor in production** - Track improvement metrics over time

For detailed testing guide: [TESTING.md](TESTING.md)

For API reference: [README.md#api-reference](README.md#-api-reference)

---

**Questions?** Check out [TESTING.md](TESTING.md) or open an issue!

