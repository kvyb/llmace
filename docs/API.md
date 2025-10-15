# LLMACE API Reference

Complete API documentation for LLMACE.

## Core Classes

### LLMACE

Main interface for the framework.

```python
from llmace import LLMACE
```

#### Constructor

```python
LLMACE(
    llm_client: Optional[OpenAI] = None,
    embedding_client: Optional[OpenAI] = None,
    config: Optional[ContextConfig] = None,
    enable_logging: bool = True
)
```

**Parameters:**
- `llm_client`: OpenAI client for reflection/curation (optional for manual mode)
- `embedding_client`: OpenAI client for embeddings (optional, defaults to llm_client)
- `config`: Configuration object (uses defaults if None)
- `enable_logging`: Enable detailed logging

#### Methods

##### `reflect()`

Reflect on an interaction and optionally update context.

```python
llmace.reflect(
    query: str,
    response: str,
    success: bool,
    feedback: Optional[str] = None,
    auto_update: bool = False,
    run_grow_and_refine: bool = True
) -> Optional[ReflectionOutput]
```

**Parameters:**
- `query`: User query or input
- `response`: System response
- `success`: Whether the interaction was successful
- `feedback`: Optional feedback about the interaction
- `auto_update`: If True, automatically apply curation
- `run_grow_and_refine`: If True, deduplicate and prune after update

**Returns:** `ReflectionOutput` object with insights and recommendations

##### `get_playbook()`

Get formatted playbook for prompt injection.

```python
playbook: str = llmace.get_playbook()
```

**Returns:** Formatted string with all context bullets organized by section

##### `add_bullet()`

Manually add a bullet to context.

```python
bullet_id: str = llmace.add_bullet(
    section: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None
)
```

**Parameters:**
- `section`: Section name (e.g., "strategies", "tips")
- `content`: Bullet content
- `metadata`: Optional metadata dictionary

**Returns:** Unique bullet ID

##### `save()` / `load()`

Persist and restore contexts.

```python
# Save
llmace.save("my_context.json")

# Load
llmace = LLMACE.load(
    "my_context.json",
    llm_client=client,
    embedding_client=client
)
```

##### `to_dict()` / `from_dict()`

Convert to/from dictionary.

```python
# Export
data = llmace.to_dict()

# Import
llmace = LLMACE.from_dict(
    data,
    llm_client=client,
    embedding_client=client
)
```

##### `grow_and_refine()`

Deduplicate and prune context.

```python
stats = llmace.grow_and_refine(
    threshold: Optional[float] = None
)
```

**Parameters:**
- `threshold`: Similarity threshold for deduplication (uses config default if None)

**Returns:** Dictionary with stats (deduped count, pruned count, etc.)

---

## Configuration

### ContextConfig

Configuration for context management.

```python
from llmace.core.schemas import ContextConfig

config = ContextConfig(
    sections=["strategies", "tips", "patterns"],
    max_bullets_per_section=20,
    dedup_threshold=0.85
)
```

**Fields:**
- `sections`: List of section names
- `max_bullets_per_section`: Maximum bullets per section (None = unlimited)
- `dedup_threshold`: Similarity threshold for deduplication (0.0-1.0)

**Defaults:**
```python
{
    "sections": ["strategies", "tips", "patterns", "warnings"],
    "max_bullets_per_section": None,
    "dedup_threshold": 0.85
}
```

---

## Data Models

### Bullet

Represents a single context item.

```python
from llmace.core.schemas import Bullet

bullet = Bullet(
    id="unique-id",
    section="strategies",
    content="Always validate input before processing",
    metadata={"source": "reflection", "timestamp": "2024-10-15"}
)
```

**Fields:**
- `id`: Unique identifier
- `section`: Section name
- `content`: Bullet text
- `metadata`: Optional metadata dictionary

### ReflectionOutput

Output from reflection.

```python
class ReflectionOutput:
    reasoning: str
    error_identification: str
    root_cause_analysis: str
    correct_approach: str
    key_insight: str
    bullet_tags: List[str]
```

### CurationOutput

Output from curation.

```python
class CurationOutput:
    reasoning: str
    deltas: List[BulletDelta]
```

### BulletDelta

Represents a context update.

```python
class BulletDelta:
    action: Literal["add", "update", "remove"]
    section: str
    content: Optional[str]
    bullet_id: Optional[str]
```

---

## Utilities

### create_embedding_function

Create embedding function from OpenAI client.

```python
from llmace.utils import create_embedding_function

embedding_fn = create_embedding_function(
    client=OpenAI(api_key="..."),
    model="text-embedding-3-small"
)

# Use it
vector = embedding_fn("some text")
```

---

## Integration Helpers

### inject_playbook_into_messages

Inject playbook into OpenAI messages.

```python
from llmace.integrations.openai import inject_playbook_into_messages

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello"}
]

updated_messages = inject_playbook_into_messages(
    messages=messages,
    playbook=llmace.get_playbook(),
    position="after_system"  # or "before_system"
)
```

---

## Examples

### Basic Usage

```python
from openai import OpenAI
from llmace import LLMACE

# Setup
client = OpenAI(api_key="...")
llmace = LLMACE(llm_client=client, embedding_client=client)

# Use in workflow
playbook = llmace.get_playbook()
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": f"You are helpful.\n\n{playbook}"},
        {"role": "user", "content": "What's 2+2?"}
    ]
)

# Learn
llmace.reflect(
    query="What's 2+2?",
    response=response.choices[0].message.content,
    success=True,
    auto_update=True
)

# Persist
llmace.save("math_agent.json")
```

### Advanced Configuration

```python
from llmace import LLMACE
from llmace.core.schemas import ContextConfig

config = ContextConfig(
    sections=["core_principles", "error_patterns", "success_patterns"],
    max_bullets_per_section=15,
    dedup_threshold=0.90
)

llmace = LLMACE(
    llm_client=llm_client,
    embedding_client=embedding_client,
    config=config,
    enable_logging=False
)
```

### Manual Mode

```python
llmace = LLMACE()  # No clients needed

# Manually curate
llmace.add_bullet("strategies", "Verify input format")
llmace.add_bullet("strategies", "Handle edge cases gracefully")

# Use immediately
playbook = llmace.get_playbook()
```

---

## Error Handling

LLMACE uses retry logic with exponential backoff for LLM calls. If all retries fail, exceptions are raised:

```python
try:
    result = llmace.reflect(query, response, success=True, auto_update=True)
except Exception as e:
    print(f"Reflection failed: {e}")
    # Handle gracefully - context unchanged
```

---

## Type Hints

LLMACE is fully typed. Use with mypy:

```python
from llmace import LLMACE
from llmace.core.schemas import ReflectionOutput

llmace: LLMACE = LLMACE(llm_client=client)
result: Optional[ReflectionOutput] = llmace.reflect(...)
```

---

## Performance Tips

1. **Batch reflections**: Don't reflect on every interaction
2. **Set max_bullets**: Prevent unbounded growth
3. **Use embeddings**: Enable semantic deduplication
4. **Persist frequently**: Save evolved contexts regularly
5. **Disable logging**: Set `enable_logging=False` for production

---

For more examples, see the [examples/](../examples/) directory.

