# Embedding Client Feature

## Overview

The ACE framework now supports passing OpenAI clients directly for embeddings, making it easier to use separate providers for LLM operations and embeddings.

## Motivation

**Problem**: OpenRouter and similar providers don't support embedding models - only OpenAI does.

**Solution**: Allow users to initialize separate OpenAI clients:
- One for LLM operations (can be OpenRouter, etc.)
- One for embeddings (must be OpenAI)

## Implementation

### New Parameter: `embedding_client`

The `ACE` class now accepts an `embedding_client` parameter:

```python
ACE(
    llm_client=None,           # For LLM operations
    embedding_client=None,      # For embeddings (NEW)
    embedding_fn=None,          # Advanced: custom function
    embedding_model="text-embedding-3-small",  # Model to use
    ...
)
```

### How It Works

1. User passes an OpenAI client via `embedding_client`
2. ACE internally converts it to an embedding function using `create_embedding_function()`
3. This function is then used for semantic deduplication

### New Utility Function

`llmace/utils/embeddings.py` provides:

```python
def create_embedding_function(
    client: Any,
    model: str = "text-embedding-3-small"
) -> Callable[[str], List[float]]:
    """Convert OpenAI client to embedding function."""
```

## Usage Examples

### Same Provider (OpenAI for both)

```python
from openai import OpenAI
from llmace import ACE

# One client for everything
client = OpenAI(api_key="sk-...")

ace = ACE(
    llm_client=client,
    embedding_client=client  # Same client
)
```

### Different Providers (OpenRouter + OpenAI)

```python
from openai import OpenAI
from llmace import ACE

# LLM via OpenRouter
llm_client = OpenAI(
    api_key="sk-or-...",
    base_url="https://openrouter.ai/api/v1"
)

# Embeddings via OpenAI
embedding_client = OpenAI(api_key="sk-...")

ace = ACE(
    llm_client=llm_client,
    embedding_client=embedding_client  # Different client
)
```

### Advanced: Custom Embedding Function

For users who need full control:

```python
from llmace import ACE

def my_embeddings(text: str) -> list[float]:
    # Custom logic here
    return embeddings

ace = ACE(embedding_fn=my_embeddings)
```

## Benefits

1. **Simplicity**: Users don't need to write embedding functions manually
2. **Flexibility**: Mix and match LLM and embedding providers
3. **Clarity**: Explicit about which client does what
4. **Backward Compatible**: `embedding_fn` still works for advanced users

## Files Modified

### Core Implementation
- `llmace/utils/embeddings.py` - New utility module
- `llmace/utils/__init__.py` - Export new functions
- `llmace/ace.py` - Accept `embedding_client` parameter
- `llmace/__init__.py` - Export `create_embedding_function`

### Examples Updated
- `examples/auto_reflection.py` - Show same client for both
- `examples/openrouter_integration.py` - Show separate clients
- `examples/README.md` - Document requirements

### Documentation Updated
- `README.md` - Updated all usage examples
- `QUICKSTART.md` - Updated quick start guide
- `PROJECT_SUMMARY.md` - Updated usage examples
- `verify_install.py` - Test new imports

## API Changes

### Constructor

**Before:**
```python
ACE(llm_client=client, embedding_fn=get_embedding)
```

**After (recommended):**
```python
ACE(llm_client=client, embedding_client=client)
```

**After (advanced):**
```python
ACE(llm_client=client, embedding_fn=custom_fn)  # Still works
```

### Load/From Dict

Both methods now accept `embedding_client`:

```python
ace = ACE.load(
    "context.json",
    llm_client=llm_client,
    embedding_client=embedding_client
)

ace = ACE.from_dict(
    data,
    llm_client=llm_client,
    embedding_client=embedding_client
)
```

## Migration Guide

### If You're Using OpenAI Only

**Old way:**
```python
client = OpenAI(api_key="...")

def get_embedding(text):
    return client.embeddings.create(...).data[0].embedding

ace = ACE(llm_client=client, embedding_fn=get_embedding)
```

**New way (simpler):**
```python
client = OpenAI(api_key="...")

ace = ACE(
    llm_client=client,
    embedding_client=client  # Much simpler!
)
```

### If You're Using OpenRouter + OpenAI

**Old way:**
```python
llm_client = OpenAI(api_key="...", base_url="https://openrouter.ai/api/v1")
embedding_client_temp = OpenAI(api_key="...")

def get_embedding(text):
    return embedding_client_temp.embeddings.create(...).data[0].embedding

ace = ACE(llm_client=llm_client, embedding_fn=get_embedding)
```

**New way (clearer):**
```python
llm_client = OpenAI(api_key="...", base_url="https://openrouter.ai/api/v1")
embedding_client = OpenAI(api_key="...")

ace = ACE(
    llm_client=llm_client,
    embedding_client=embedding_client  # Clear separation!
)
```

## Testing

Run the updated examples to verify:

```bash
# Test with OpenAI (same client)
export OPENAI_API_KEY='...'
python examples/auto_reflection.py

# Test with OpenRouter + OpenAI (separate clients)
export OPENROUTER_API_KEY='...'
export OPENAI_API_KEY='...'
python examples/openrouter_integration.py

# Verify installation
python verify_install.py
```

## Summary

This feature makes ACE easier to use by:
- Eliminating boilerplate embedding function code
- Making provider separation explicit and clear
- Maintaining backward compatibility
- Providing sensible defaults

Users can now pass OpenAI clients directly, and ACE handles the conversion internally.

