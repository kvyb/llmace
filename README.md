# 🎯 LLMace

[![PyPI version](https://badge.fury.io/py/llmace.svg)](https://badge.fury.io/py/llmace)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Agentic Context Engineering** - A Python framework for evolving LLM contexts through reflection and curation.

> 📄 Based on research: [Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models](https://arxiv.org/abs/2510.04618)  
> _Qizheng Zhang, Changran Hu, et al. (Stanford University & SambaNova Systems)_

## What is LLMACE?

LLMACE transforms static prompts into **living playbooks** that learn and improve over time. Instead of manually refining prompts, LLMACE:
- ✅ **Reflects** on task performance
- ✅ **Curates** insights into reusable strategies  
- ✅ **Grows** context with new knowledge
- ✅ **Refines** to prevent bloat through deduplication and pruning

Perfect for **agents**, **reasoning tasks**, **Q&A systems**, and any multi-turn LLM workflow.

---

## 🚀 Quick Start

```bash
pip install llmace
```

```python
from openai import OpenAI
from llmace import LLMACE

# Initialize clients
llm_client = OpenAI(api_key="your-api-key")
embedding_client = OpenAI(api_key="your-api-key")

# Create LLMACE instance
llmace = LLMACE(
    llm_client=llm_client,
    embedding_client=embedding_client
)

# Use in your workflow
playbook = llmace.get_playbook()  # Inject into your system prompt
response = your_llm_call(playbook + user_query)

# Learn from the interaction
llmace.reflect(
    query=user_query,
    response=response,
    success=True,  # or False if it failed
    auto_update=True
)

# Save for next session
llmace.save("my_playbook.json")
```

---

## 🌟 Key Features

| Feature | Description |
|---------|-------------|
| **🧠 Auto-Learning** | Automatically extracts insights from interactions |
| **🔄 Modular Design** | Separate components for reflection, curation, refinement |
| **🌐 Universal** | Works across all domains and LLM providers |
| **🔌 Easy Integration** | Drop-in replacement for static prompts |
| **🧹 Smart Deduplication** | Embedding-based semantic similarity detection |
| **💾 Serialization** | Save/load evolved contexts between sessions |
| **⚙️ Configurable** | Control growth limits, dedup thresholds, sections |

---

## 📦 Installation

### Basic Installation
```bash
pip install llmace
```

### With Optional Dependencies

**Embeddings** (recommended for deduplication):
```bash
pip install llmace[embeddings]
```

**Agent Frameworks** (LangGraph support):
```bash
pip install llmace[agents]
```

**All Features**:
```bash
pip install llmace[all]
```

**Development**:
```bash
pip install llmace[dev]
```

### Install from Source

**Using uv** (recommended for development):
```bash
# Clone the repository
git clone https://github.com/llmace/llmace.git
cd llmace

# Create virtual environment and install with uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[all]"
```

**Using pip**:
```bash
git clone https://github.com/llmace/llmace.git
cd llmace
pip install -e ".[all]"
```

---

## 🔧 Configuration

### OpenRouter + OpenAI (Recommended)
Best of both worlds: 100+ models via OpenRouter, quality embeddings from OpenAI.

```python
from openai import OpenAI

llm_client = OpenAI(
    api_key="sk-or-v1-...",
    base_url="https://openrouter.ai/api/v1"
)
embedding_client = OpenAI(api_key="sk-...")
```

### Environment Variables
Create a `.env` file:
```env
OPENROUTER_API_KEY=sk-or-v1-your-key
OPENAI_API_KEY=sk-your-key
```

**Priority Logic:**
- If `OPENROUTER_API_KEY` → LLM via OpenRouter
- If `OPENAI_API_KEY` → Embeddings via OpenAI (recommended)
- Single key → Use for both

---

## 📖 Usage

### Manual Mode (Bring Your Own Insights)
```python
from llmace import LLMACE

llmace = LLMACE()  # No LLM client needed

# Manually add insights
llmace.add_bullet(
    section="strategies",
    content="Always verify input data before processing"
)

# Use in your prompt
playbook = llmace.get_playbook()
```

### Automatic Mode (LLM-Driven Reflection)
```python
llmace = LLMACE(
    llm_client=llm_client,
    embedding_client=embedding_client
)

# Let LLMACE reflect automatically
llmace.reflect(
    query="Calculate tax for $1000 purchase",
    response="Tax is $80",
    success=True,  # Just indicate if it worked
    auto_update=True  # Automatically extract and add insights
)

# Optional: provide feedback for additional context
llmace.reflect(
    query="Complex calculation failed",
    response="Error: division by zero",
    success=False,
    feedback="Input validation was missing",  # Extra context for LLM
    auto_update=True
)
```

### Advanced Configuration
```python
from llmace import LLMACE
from llmace.core.schemas import ContextConfig

config = ContextConfig(
    max_bullets_per_section=20,  # Limit growth
    dedup_threshold=0.85,        # Semantic similarity threshold
    sections=["strategies", "patterns", "edge_cases"]  # Custom sections
)

llmace = LLMACE(
    llm_client=llm_client,
    embedding_client=embedding_client,
    config=config
)
```

### Persistence
```python
# Save evolved context
llmace.save("my_agent_playbook.json")

# Load in next session
llmace = LLMACE.load(
    "my_agent_playbook.json",
    llm_client=llm_client,
    embedding_client=embedding_client
)
```

### Integration with LangGraph
```python
from langgraph.prebuilt import create_react_agent
from llmace import LLMACE

llmace = LLMACE(llm_client=client, embedding_client=client)

# Inject playbook into system prompt
system_prompt = f"""You are a helpful assistant.

{llmace.get_playbook()}

Use the strategies above to guide your responses."""

agent = create_react_agent(model, tools, state_modifier=system_prompt)

# After each turn, reflect
result = agent.invoke({"messages": [("user", query)]})
llmace.reflect(
    query=query,
    response=result["messages"][-1].content,
    success=True,
    auto_update=True
)
```

---

## 📚 Documentation

- **[Quick Start Guide](docs/QUICKSTART.md)** - Get up and running in 5 minutes
- **[Setup Guide](docs/SETUP_GUIDE.md)** - Detailed configuration instructions
- **[Testing Guide](docs/TESTING.md)** - Run benchmarks and tests
- **[Examples](examples/)** - Real-world usage examples
- **[API Reference](docs/API.md)** - Full API documentation

---

## 🧪 Testing

LLMACE includes comprehensive benchmarks:

```bash
# Quick functionality test
python tests/quick_test.py

# Core unit tests
python tests/test_core.py

# Agentic benchmark (LangGraph)
python tests/benchmark_suite.py

# FAQ learning benchmark
python tests/benchmark_faq.py

# LangGraph integration example
python tests/test_langgraph_integration.py
```

**Example Results:**
```
Baseline:  65% success rate
LLMACE:    87% success rate (+22% improvement)
```

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Quick Contribution Steps:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🎓 Citation

If you use LLMACE in your research, please cite the original paper:

```bibtex
@article{zhang2024ace,
  title={Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models},
  author={Zhang, Qizheng and Hu, Changran and Upasani, Shubhangi and Ma, Boyuan and Hong, Fenglu and Kamanuru, Vamsidhar and Rainton, Jay and Wu, Chen and Ji, Mengmeng and Li, Hanchen and Thakker, Urmish and Zou, James and Olukotun, Kunle},
  journal={arXiv preprint arXiv:2510.04618},
  year={2024}
}
```

---

## 🔗 Links

- **PyPI**: https://pypi.org/project/llmace/
- **GitHub**: https://github.com/llmace/llmace
- **Issues**: https://github.com/llmace/llmace/issues

---

## 🙏 Acknowledgments

Built based on research by Qizheng Zhang, Changran Hu, and colleagues at Stanford University and SambaNova Systems. 

**Related Projects:**
- [ACE-open](https://github.com/sci-m-wang/ACE-open) - Alternative research-focused implementation of the ACE paper

Special thanks to the open-source community.

---

**Star ⭐ this repo if LLMACE helps your project!**
