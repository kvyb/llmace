# LLMACE Tests

This directory contains tests and benchmarks for LLMACE.

## Test Files

### Core Tests

**`test_core.py`**
- Unit tests for core LLMACE functionality
- Tests context management, bullet operations, serialization
- Run: `python tests/test_core.py`

### Benchmarks

**`benchmark_suite.py`** ⭐ **Recommended**
- Comprehensive agentic flow benchmark using LangGraph
- Tests LLMACE on realistic multi-step tasks (bill splitting, file operations, etc.)
- Compares Baseline vs LLMACE vs ICL
- Uses LLM-as-a-judge for evaluation
- Run: `python tests/benchmark_suite.py`
- Expected: +20-30% improvement over baseline

**`benchmark_faq.py`**
- Tests learning progression on FAQ-style questions
- Organized by semantic clusters
- Tracks correctness improvement over time
- Run: `python tests/benchmark_faq.py`
- Note: FAQ tests show modest gains (fact recall vs reasoning)

### Integration Tests

**`test_langgraph_integration.py`**
- Example integration with LangGraph framework
- Shows how to inject LLMACE playbook into agent system prompt
- Demonstrates reflection after agent turns
- Run: `python tests/test_langgraph_integration.py`

### Quick Tests

**`quick_test.py`**
- Fast sanity check for LLMACE functionality
- Tests basic reflection and learning
- Run: `python tests/quick_test.py`
- Takes ~2 minutes

## Running Tests

### Individual Tests
```bash
# Quick sanity check
python tests/quick_test.py

# Core unit tests
python tests/test_core.py

# Agentic benchmark (recommended for performance evaluation)
python tests/benchmark_suite.py

# FAQ learning benchmark
python tests/benchmark_faq.py

# LangGraph integration example
python tests/test_langgraph_integration.py
```

### Using Test Runner Script
```bash
# Run all tests with menu
./run_tests.sh

# Or select specific tests interactively
```

## Environment Setup

Tests require API keys configured in `.env`:

```env
# Required
OPENAI_API_KEY=sk-...          # For embeddings
OPENROUTER_API_KEY=sk-or-v1-...  # For LLM calls

# Optional
TEST_MODEL=google/gemini-2.5-flash  # Model for tests
JUDGE_MODEL=google/gemini-2.5-flash  # Model for evaluation
LLMACE_VERBOSE=false            # Enable verbose logging
```

See `env.example` for full configuration options.

## Test Requirements

Install test dependencies:
```bash
pip install llmace[all]  # Includes langgraph, langchain
```

Or manually:
```bash
pip install langgraph langchain-openai langchain-core
```

## Expected Results

### Benchmark Suite (Agentic Tasks)
```
Method          Success Rate    Avg Score
----------------------------------------------
Baseline        65%            0.653
LLMACE          87%            0.870  (+22%)
ICL             78%            0.782
```

### FAQ Benchmark (Learning Progression)
```
Stage           Baseline    LLMACE    Improvement
-------------------------------------------------
Early (Q1-20)   80%         85%       +5%
Mid (Q21-35)    90%         93%       +3%
Late (Q36-50)   85%         88%       +3%
```

## Performance Notes

- **Agentic tasks**: LLMACE shows significant gains (+20-30%)
- **FAQ/recall tasks**: Modest improvements (+3-5%)
- **Best for**: Multi-step reasoning, decision-making, edge case handling
- **Less effective for**: Simple fact lookup, single-step Q&A

## Troubleshooting

**API Key Issues:**
```bash
# Check keys are loaded
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('OPENROUTER_API_KEY'))"
```

**Model Not Found:**
- Ensure model names match OpenRouter catalog
- Default fallback: `google/gemini-2.5-flash`
- Check https://openrouter.ai/models

**Import Errors:**
```bash
# Reinstall with all dependencies
pip install -e .[all]
```

## Contributing Tests

When adding new tests:
1. Follow naming convention: `test_*.py` or `benchmark_*.py`
2. Include docstring explaining test purpose
3. Add entry to this README
4. Update `run_tests.sh` if adding new benchmark
5. Document expected results

## More Information

See [docs/TESTING.md](../docs/TESTING.md) for detailed testing guide.

