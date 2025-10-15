# Quick Start Guide

Get up and running with ACE in 5 minutes!

## Installation

```bash
pip install llmace
```

## Your First ACE Program

```python
from llmace import ACE

# 1. Create an ACE instance
ace = ACE()

# 2. Manually add some knowledge
ace.add_bullet(
    section="strategies",
    content="Break complex problems into smaller steps"
)

ace.add_bullet(
    section="best_practices",
    content="Always validate inputs before processing"
)

# 3. Get the playbook
playbook = ace.get_playbook()
print(playbook)

# 4. Save for later
ace.save("my_first_context.json")
```

**Output:**
```
**Strategies**

- Break complex problems into smaller steps

**Best Practices**

- Always validate inputs before processing
```

## Using with OpenAI

```python
from llmace import ACE
from openai import OpenAI

# Initialize client
client = OpenAI(api_key="your-key")

# Initialize ACE with LLM and embedding support
ace = ACE(
    llm_client=client,
    embedding_client=client  # Same client for embeddings
)

# Generate response with context
messages = [
    {"role": "system", "content": ace.get_playbook()},
    {"role": "user", "content": "Help me debug this code..."}
]

response = client.chat.completions.create(
    model="gpt-4",
    messages=messages
)

# Learn from the interaction
ace.reflect(
    query="Help me debug this code...",
    response=response.choices[0].message.content,
    success=True,
    feedback="Solution worked perfectly"
)

# Playbook automatically evolves!
```

## Three Ways to Use ACE

### 1. Manual Mode (No LLM Required)

Perfect for when you want full control:

```python
ace = ACE()

reflection = {
    "reasoning": "Failed because of X",
    "key_insight": "Always do Y before Z",
    "error_identification": None,
    "root_cause_analysis": None,
    "correct_approach": None,
    "bullet_tags": []
}

curation = {
    "reasoning": "Adding insight about Y",
    "deltas": [{
        "operation": "add",
        "section": "strategies",
        "content": "Always do Y before Z",
        "metadata": {}
    }]
}

ace.reflect(
    query="...",
    response="...",
    success=False,
    reflection_result=reflection,
    curation_result=curation
)
```

### 2. Auto Mode (LLM-Driven)

Let the LLM analyze and learn:

```python
from openai import OpenAI

client = OpenAI(api_key="your-key")
ace = ACE(llm_client=client)

# Just call reflect - ACE handles the rest
ace.reflect(
    query="Calculate 15 * 24",
    response="360",
    success=True
)
```

### 3. Hybrid Mode

Use LLM for some tasks, manual for others:

```python
ace = ACE(llm_client=client)

# Auto reflection
ace.reflect(query="...", response="...", success=True)

# Manual bullet addition
ace.add_bullet(section="strategies", content="Custom strategy")

# Manual grow-and-refine
ace.grow_and_refine()
```

## Key Concepts in 30 Seconds

**Bullets**: Individual pieces of knowledge
```python
bullet = ace.add_bullet(section="strategies", content="Do X")
```

**Sections**: Categories for organization
- strategies, insights, common_mistakes, best_practices, patterns

**Playbook**: Formatted output for prompts
```python
playbook = ace.get_playbook()
```

**Reflection**: Analyzing what happened
```python
ace.reflect(query="...", response="...", success=True)
```

**Grow-and-Refine**: Cleanup and deduplication
```python
ace.grow_and_refine()
```

## Common Patterns

### Pattern 1: Static Playbook

```python
# Build once, use many times
ace = ACE()
ace.add_bullet(section="strategies", content="Strategy 1")
ace.add_bullet(section="strategies", content="Strategy 2")
ace.save("playbook.json")

# Later, in your app
ace = ACE.load("playbook.json")
playbook = ace.get_playbook()
```

### Pattern 2: Evolving Agent

```python
ace = ACE(llm_client=client)

for task in tasks:
    # Use current playbook
    response = generate_with_playbook(task, ace.get_playbook())
    
    # Learn from execution
    ace.reflect(
        query=task,
        response=response,
        success=evaluate(response)
    )
    
    # Playbook gets better over time!
```

### Pattern 3: Domain Expert

```python
# Pre-load domain knowledge
ace = ACE()

for rule in domain_rules:
    ace.add_bullet(section="strategies", content=rule)

for mistake in common_mistakes:
    ace.add_bullet(section="common_mistakes", content=mistake)

# Use in production
playbook = ace.get_playbook()
```

## Next Steps

1. **Try the examples**: `python examples/basic_usage.py`
2. **Read the full README**: [README.md](README.md)
3. **Check the API docs**: [API Reference](README.md#api-reference)
4. **Join the community**: [Discussions](https://github.com/yourusername/llmace/discussions)

## Need Help?

- 📖 [Full Documentation](README.md)
- 💬 [GitHub Discussions](https://github.com/yourusername/llmace/discussions)
- 🐛 [Report Issues](https://github.com/yourusername/llmace/issues)
- 📧 Email: support@example.com

---

Happy context engineering! 🎯

