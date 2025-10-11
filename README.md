# 🎯 LLMace - Agentic Context Engineering

**ACE (Agentic Context Engineering)** is a Python framework for building and evolving comprehensive contexts in LLM workflows. Instead of static prompts, ACE treats contexts as living playbooks that accumulate strategies, insights, and best practices over time through reflection and curation.

Based on the research paper: [Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models](https://arxiv.org/abs/XXXX.XXXXX)

## 🌟 Key Features

- **Evolving Playbooks**: Contexts grow and refine over time, accumulating domain knowledge
- **Modular Architecture**: Separate components for generation, reflection, and curation
- **Universal Design**: Works across domains - agents, reasoning tasks, QA systems
- **Flexible Integration**: Plugs into any OpenAI-compatible LLM workflow
- **Semantic Deduplication**: Prevents redundancy using embedding-based similarity
- **Two Modes**: Automatic (LLM-driven) or manual (bring your own insights)
- **Serialization**: Easy save/load of evolved contexts

## 📦 Installation

```bash
pip install llmace
```

For semantic deduplication with embeddings:
```bash
pip install llmace[embeddings]
```

For development:
```bash
pip install llmace[dev]
```

## 🚀 Quick Start

### Basic Usage

```python
from llmace import ACE
from openai import OpenAI

# Initialize ACE with an LLM client
client = OpenAI(api_key="your-api-key")
ace = ACE(
    llm_client=client,
    embedding_client=client  # Same client for embeddings
)

# Get the playbook for prompt injection
playbook = ace.get_playbook()

# Use playbook in your prompts
messages = [
    {"role": "system", "content": f"You are a helpful assistant.\n\n{playbook}"},
    {"role": "user", "content": "Help me solve this problem..."}
]
response = client.chat.completions.create(model="gpt-4", messages=messages)

# Reflect on the execution to evolve the context
ace.reflect(
    query="Help me solve this problem...",
    response=response.choices[0].message.content,
    success=True,  # or False if it failed
    feedback="The solution was correct but could be more efficient"
)

# Save evolved context
ace.save("my_context.json")
```

### Using with OpenRouter

```python
from openai import OpenAI
from llmace import ACE

# Initialize OpenAI client with OpenRouter for LLM
llm_client = OpenAI(
    api_key="your-openrouter-key",
    base_url="https://openrouter.ai/api/v1"
)
llm_client.default_model = "anthropic/claude-3.5-sonnet"

# Separate OpenAI client for embeddings (OpenRouter doesn't support embeddings)
embedding_client = OpenAI(api_key="your-openai-key")

# Use with ACE
ace = ACE(
    llm_client=llm_client,
    embedding_client=embedding_client
)
```

### Manual Mode (No LLM Required for Reflection)

```python
from llmace import ACE

# Initialize without LLM client
ace = ACE()

# Manually provide reflection results
reflection_result = {
    "reasoning": "The approach failed because...",
    "key_insight": "Always validate input before processing",
    "error_identification": "Missing input validation",
    "root_cause_analysis": "...",
    "correct_approach": "...",
    "bullet_tags": []
}

curation_result = {
    "reasoning": "Adding validation strategy...",
    "deltas": [
        {
            "operation": "add",
            "section": "strategies",
            "content": "Always validate inputs before processing to avoid runtime errors",
            "metadata": {}
        }
    ]
}

ace.reflect(
    query="...",
    response="...",
    success=False,
    reflection_result=reflection_result,
    curation_result=curation_result
)
```

### With Semantic Deduplication

Embeddings enable automatic deduplication of similar bullets:

```python
from llmace import ACE
from openai import OpenAI

# Initialize clients
client = OpenAI(api_key="your-api-key")

# Initialize ACE with embedding client
ace = ACE(
    llm_client=client,
    embedding_client=client  # ACE will use this for semantic deduplication
)

# Deduplication happens automatically during grow-and-refine
stats = ace.grow_and_refine()
print(f"Removed {stats['deduped']} duplicate bullets")
```

**Advanced: Custom embedding function**
```python
from llmace import ACE

# Define custom embedding function
def get_embedding(text: str) -> list[float]:
    # Your custom embedding logic
    return embeddings

ace = ACE(embedding_fn=get_embedding)  # Use custom function
```

## 🎓 Core Concepts

### Bullets
Atomic units of knowledge with:
- **ID**: Unique identifier
- **Section**: Category (strategies, insights, common_mistakes, etc.)
- **Content**: The actual knowledge
- **Counters**: Helpful/harmful feedback counts
- **Metadata**: Additional context

### Sections
Organize bullets into categories:
- `strategies`: High-level approaches
- `insights`: Key lessons learned
- `common_mistakes`: Pitfalls to avoid
- `best_practices`: Proven methods
- `patterns`: Recurring solutions

### Grow-and-Refine
Periodic maintenance that:
1. Deduplicates similar bullets using semantic similarity
2. Prunes bullets with negative scores
3. Enforces section size limits

## 📚 Advanced Usage

### Custom Configuration

```python
from llmace import ACE, ContextConfig

config = ContextConfig(
    sections=["tactics", "rules", "examples"],
    dedup_threshold=0.90,  # Higher = stricter deduplication
    max_bullets_per_section=50,
    enable_deduplication=True,
    prune_negative_bullets=True
)

ace = ACE(config=config)
```

### Custom Prompt Templates

```python
custom_reflection_prompt = """
Analyze this execution:
Query: {query}
Response: {response}
Success: {success}

Provide insights in JSON format...
"""

ace = ACE(
    llm_client=client,
    reflection_prompt=custom_reflection_prompt
)
```

### Direct Context Manipulation

```python
# Add bullets manually
bullet_id = ace.add_bullet(
    section="strategies",
    content="Use chain-of-thought reasoning for complex problems"
)

# Access underlying context
print(f"Total bullets: {len(ace.context)}")
bullets = ace.context.get_bullets_by_section("strategies")
```

### Integration with OpenAI Messages

```python
from llmace.integrations import inject_playbook_into_messages

messages = [
    {"role": "user", "content": "Help me with this task"}
]

# Inject playbook into messages
enhanced_messages = inject_playbook_into_messages(
    messages=messages,
    context=ace.context,
    position="system"  # or "before_user", "after_system"
)
```

## 🔧 API Reference

### ACE Class

**Constructor**
```python
ACE(
    llm_client=None,           # OpenAI-compatible client for LLM
    embedding_fn=None,          # Function: str -> list[float] (advanced)
    embedding_client=None,      # OpenAI client for embeddings
    embedding_model="text-embedding-3-small",  # Model for embedding_client
    config=None,                # ContextConfig object
    context=None,               # Existing ACEContext
    reflection_prompt=None,     # Custom reflection template
    curation_prompt=None,       # Custom curation template
    enable_logging=False        # Enable logging
)
```

**Methods**
- `get_playbook()`: Get formatted playbook string
- `reflect()`: Reflect on execution and update context
- `add_bullet()`: Manually add a bullet
- `grow_and_refine()`: Run deduplication and pruning
- `save(filepath)`: Save context to JSON
- `load(filepath)`: Load context from JSON

## 📖 Examples

See the `examples/` directory for complete working examples:
- `basic_usage.py`: Simple reflection workflow
- `auto_reflection.py`: Automatic reflection with OpenAI
- `openrouter_integration.py`: Using OpenRouter as LLM provider

## 🧪 Development

```bash
# Clone the repository
git clone https://github.com/yourusername/llmace.git
cd llmace

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black llmace/
ruff check llmace/
```

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

Based on the ACE framework from the paper "Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models" by Zhang et al.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📮 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/llmace/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/llmace/discussions)

