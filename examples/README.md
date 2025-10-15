# ACE Examples

This directory contains working examples demonstrating different use cases of the ACE framework.

## Examples

### 1. `basic_usage.py` - Manual Reflection Mode

Demonstrates using ACE without automatic LLM calls. You manually provide reflection and curation results.

**Use case**: When you want full control over the reflection process or don't want to incur LLM API costs.

**Run:**
```bash
python basic_usage.py
```

**Key concepts demonstrated:**
- Manual reflection and curation
- Adding bullets to context
- Formatting playbooks
- Saving and loading contexts

---

### 2. `auto_reflection.py` - Automatic Reflection with OpenAI

Shows how to use ACE with automatic LLM-driven reflection and curation using OpenAI.

**Use case**: When you want the LLM to automatically analyze executions and extract insights.

**Requirements:**
- OpenAI API key

**Setup:**
```bash
export OPENAI_API_KEY='your-openai-api-key'
```

**Run:**
```bash
python auto_reflection.py
```

**Key concepts demonstrated:**
- Automatic reflection mode
- Multi-turn learning
- Playbook injection into prompts
- Context evolution over multiple tasks

---

### 3. `openrouter_integration.py` - Using OpenRouter

Demonstrates using ACE with OpenRouter to access various LLM providers (Claude, GPT-4, Llama, etc.).

**Use case**: When you want to use different LLM providers through a single API.

**Requirements:**
- OpenRouter API key (for LLM)
- OpenAI API key (for embeddings - optional)

**Setup:**
```bash
export OPENROUTER_API_KEY='your-openrouter-api-key'
export OPENAI_API_KEY='your-openai-api-key'  # Optional, for embeddings
```

**Run:**
```bash
python openrouter_integration.py
```

**Key concepts demonstrated:**
- OpenRouter integration for LLM
- Separate OpenAI client for embeddings
- Using different providers for LLM vs embeddings
- Context persistence and reuse

---

## Common Patterns

### Getting a Playbook for Prompt Injection

```python
from llmace import LLMACE

llmace = LLMACE()
playbook = ace.get_playbook()

# Use in your prompts
messages = [
    {"role": "system", "content": f"Instructions...\n\n{playbook}"},
    {"role": "user", "content": "Your query"}
]
```

### Reflecting on Execution

```python
# After task execution
result = ace.reflect(
    query="Original task",
    response="Generated response",
    success=True,  # or False
    feedback="Optional feedback",
    auto_update=True  # Automatically update context
)
```

### Saving and Loading Context

```python
# Save
ace.save("my_context.json")

# Load
llmace = LLMACE.load("my_context.json", llm_client=client)
```

## Tips

1. **Start with manual mode** (`basic_usage.py`) to understand the concepts
2. **Use automatic mode** for production workflows where you want continuous learning
3. **Enable embeddings** for better deduplication (requires `pip install llmace[embeddings]`)
4. **Save contexts regularly** to preserve learned knowledge
5. **Customize sections** based on your domain needs

## Troubleshooting

**"No API key found"**
- Make sure you've exported the required environment variables
- Check the variable name matches exactly (OPENAI_API_KEY or OPENROUTER_API_KEY)

**"Deduplication not working"**
- Ensure you've provided an `embedding_fn` to ACE
- Install embeddings dependencies: `pip install llmace[embeddings]`

**"Context not evolving"**
- Check that `auto_update=True` in `reflect()`
- Verify LLM client is properly configured
- Enable logging to see what's happening: `ACE(enable_logging=True)`

