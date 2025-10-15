# ACE Implementation Overview

This document provides a comprehensive overview of the ACE (Agentic Context Engineering) implementation.

## Architecture

### Core Components

#### 1. Data Layer (`llmace/core/`)

**`schemas.py`** - Pydantic models for data validation:
- `Bullet`: Atomic knowledge unit with metadata
- `BulletDelta`: Represents proposed changes
- `ContextConfig`: Configuration with validation
- `ReflectionInput`: Input for reflection process
- `ReflectionOutput`: Output from reflection
- `CurationOutput`: Delta updates from curation

**`context.py`** - Context management:
- `ACEContext`: Main class managing bullet collections
- Methods: `add_bullet()`, `merge_delta()`, `apply_bullet_tags()`, etc.
- Serialization: `to_dict()`, `from_dict()`
- Section-based organization

**`operations.py`** - Advanced operations:
- `ContextOperations`: Deduplication and refinement
- Semantic similarity using embeddings
- Grow-and-refine process
- Pruning strategies

#### 2. Reflection Layer (`llmace/reflection/`)

**`reflector.py`** - Analyzes executions:
- `Reflector`: Extracts insights from task executions
- Two modes: automatic (LLM-driven) or manual (user-provided)
- Configurable prompt templates

**`curator.py`** - Generates updates:
- `Curator`: Converts insights into delta updates
- Avoids redundancy by checking existing context
- Organizes insights by section

#### 3. Integration Layer (`llmace/integrations/`)

**`openai.py`** - OpenAI compatibility:
- `inject_playbook_into_messages()`: Helper function
- `ACEOpenAIWrapper`: Wrapper class for automatic injection
- Support for OpenAI-compatible APIs (OpenRouter, etc.)

#### 4. Configuration (`llmace/config/`)

**`defaults.py`** - Default settings:
- Universal sections for multi-turn workflows
- Default configuration parameters

**`templates.py`** - Prompt templates:
- Reflection prompt for analyzing executions
- Curation prompt for extracting insights
- Generator instruction template

#### 5. Utilities (`llmace/utils/`)

**`formatting.py`** - Playbook formatting:
- `PlaybookFormatter`: Converts context to structured strings
- Customizable templates
- Metadata inclusion options

**`logging.py`** - Logging setup:
- Configurable logging for debugging

#### 6. Main Interface (`llmace/ace.py`)

**`ACE`** - High-level API:
- Combines all components
- Simple interface for common tasks
- Handles both automatic and manual modes

## Data Flow

### 1. Initialization
```
User → ACE(config, llm_client, embedding_fn)
     → ACEContext created
     → ContextOperations initialized
     → Reflector/Curator initialized
```

### 2. Playbook Generation
```
ACE.get_playbook()
  → ACEContext.get_all_bullets()
  → PlaybookFormatter.format_playbook()
  → Structured string output
```

### 3. Reflection Process
```
ACE.reflect(query, response, success)
  → ReflectionInput created
  → Reflector.reflect()
    → [Auto mode: LLM call for analysis]
    → [Manual mode: use provided result]
  → ReflectionOutput
  → Apply bullet tags to context
  → Curator.curate()
    → [Auto mode: LLM call for curation]
    → [Manual mode: use provided result]
  → CurationOutput with deltas
  → ACEContext.merge_deltas()
  → ContextOperations.grow_and_refine()
  → Updated context
```

### 4. Serialization
```
ACE.save(filepath)
  → ACEContext.to_dict()
  → JSON.dump()

ACE.load(filepath)
  → JSON.load()
  → ACEContext.from_dict()
  → ACE instance restored
```

## Key Design Decisions

### 1. Pydantic for Validation
- Type safety and validation
- Automatic serialization
- Clear error messages

### 2. Separation of Concerns
- Core: Data structures and operations
- Reflection: Analysis and curation
- Integration: LLM client compatibility
- Each layer has minimal dependencies

### 3. Two-Mode Operation
- **Automatic**: LLM-driven, minimal user code
- **Manual**: Full control, no LLM required
- Allows testing without API costs

### 4. OpenAI Compatibility
- Standard interface works with many providers
- Easy to extend for other APIs
- Examples for common providers

### 5. Semantic Deduplication
- Optional (requires embedding function)
- Prevents redundant knowledge
- Configurable threshold

### 6. Incremental Updates
- Delta-based changes (not full rewrites)
- Preserves existing knowledge
- Efficient for large contexts

## Usage Patterns

### Pattern 1: Static Knowledge Base
```python
ace = ACE()
# Add domain knowledge
for rule in rules:
    ace.add_bullet(section="strategies", content=rule)
ace.save("domain_kb.json")

# Later: load and use
ace = ACE.load("domain_kb.json")
playbook = ace.get_playbook()
```

### Pattern 2: Online Learning
```python
ace = ACE(llm_client=client)

for task in stream:
    # Use current knowledge
    response = generate(task, ace.get_playbook())
    
    # Learn from execution
    ace.reflect(query=task, response=response, success=check(response))
    
    # Context evolves automatically
```

### Pattern 3: Offline Training
```python
ace = ACE(llm_client=client, embedding_fn=embed)

# Train on dataset
for example in training_data:
    ace.reflect(
        query=example.input,
        response=example.output,
        success=True,
        ground_truth=example.target
    )

# Periodic cleanup
if len(ace.context) > 1000:
    ace.grow_and_refine()

# Deploy trained context
ace.save("trained_context.json")
```

### Pattern 4: A/B Testing
```python
# Baseline
ace_baseline = ACE()
ace_baseline.add_bullet(section="strategies", content="Basic strategy")

# Variant
ace_variant = ACE.load("evolved_context.json")

# Compare
results_baseline = run_tasks(ace_baseline)
results_variant = run_tasks(ace_variant)
```

## Extension Points

### Custom LLM Clients
```python
class MyLLMClient:
    def chat_completions_create(self, messages, **kwargs):
        # Your implementation
        pass

ace = ACE(llm_client=MyLLMClient())
```

### Custom Embedding Functions
```python
def my_embeddings(text: str) -> list[float]:
    # Your implementation
    return embeddings

ace = ACE(embedding_fn=my_embeddings)
```

### Custom Prompt Templates
```python
custom_reflection = "..."
custom_curation = "..."

ace = ACE(
    reflection_prompt=custom_reflection,
    curation_prompt=custom_curation
)
```

### Custom Sections
```python
config = ContextConfig(
    sections=["domain_rules", "examples", "edge_cases"],
    dedup_threshold=0.9
)

ace = ACE(config=config)
```

## Performance Considerations

### Memory
- Context size grows with bullets
- Use `max_bullets_per_section` to limit
- Regular `grow_and_refine()` recommended

### API Costs
- Auto-reflection calls LLM twice per task
- Use manual mode for cost control
- Batch reflections when possible

### Latency
- Deduplication with embeddings adds overhead
- Disable if not needed
- Cache embeddings when possible

### Storage
- JSON serialization is human-readable
- For large contexts, consider compression
- Periodic backups recommended

## Testing Strategy

### Unit Tests
- Test each component in isolation
- Mock LLM clients
- Test edge cases

### Integration Tests
- Test component interactions
- Test serialization round-trips
- Test with real LLM clients (optional)

### Example Tests
- Run all examples successfully
- Verify outputs are reasonable
- Check for errors/warnings

## Future Enhancements

### Potential Additions
- Additional storage backends (Redis, PostgreSQL)
- More sophisticated deduplication
- Visualization tools
- CLI tools
- Integration with popular frameworks

### Research Directions
- Multi-epoch adaptation strategies
- Automatic section discovery
- Cross-domain transfer
- Collaborative context evolution

## Troubleshooting

### Common Issues

**Import errors:**
- Ensure package is installed: `pip install -e .`
- Check Python version (>=3.9)

**LLM not responding:**
- Verify API key is set
- Check internet connection
- Try with manual mode first

**Deduplication not working:**
- Ensure embedding function is provided
- Check embedding dimensions match
- Verify threshold is reasonable (0.8-0.95)

**Context not evolving:**
- Check `auto_update=True`
- Verify LLM client is configured
- Enable logging to debug

## Summary

ACE provides a production-ready framework for evolving contexts in LLM workflows. The implementation balances:
- **Flexibility**: Multiple modes and extension points
- **Usability**: Simple API for common cases
- **Reliability**: Type-safe with validation
- **Performance**: Efficient operations and caching
- **Maintainability**: Clean architecture and documentation

The modular design allows using components independently or together, making it suitable for a wide range of applications from research prototypes to production systems.

