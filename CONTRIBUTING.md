# Contributing to LLMace

Thank you for your interest in contributing to ACE (Agentic Context Engineering)! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/llmace.git
cd llmace
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install in development mode:**
```bash
pip install -e ".[dev,embeddings]"
```

## Project Structure

```
llmace/
├── llmace/              # Main package
│   ├── core/           # Core context management
│   ├── reflection/     # Reflection and curation
│   ├── integrations/   # LLM client integrations
│   ├── config/         # Default configs and templates
│   └── utils/          # Utilities
├── examples/           # Usage examples
├── tests/              # Test suite
└── docs/              # Documentation
```

## Code Style

We use standard Python tools for code quality:

- **Black** for code formatting
- **Ruff** for linting
- **MyPy** for type checking

Run before committing:
```bash
black llmace/ tests/ examples/
ruff check llmace/ tests/ examples/
mypy llmace/
```

## Testing

We use pytest for testing:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=llmace --cov-report=html

# Run specific test file
pytest tests/test_core.py

# Run specific test
pytest tests/test_core.py::TestBullet::test_bullet_creation
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Use descriptive test names
- Include docstrings explaining what's being tested
- Aim for >80% code coverage

Example:
```python
def test_feature_name():
    """Test that feature works as expected."""
    # Arrange
    context = ACEContext()
    
    # Act
    result = context.add_bullet(section="test", content="Test")
    
    # Assert
    assert result is not None
    assert result.content == "Test"
```

## Pull Request Process

1. **Fork the repository** and create a new branch:
```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes** following the code style guidelines

3. **Add tests** for any new functionality

4. **Update documentation** if needed (README, docstrings, etc.)

5. **Run tests and linters:**
```bash
pytest
black llmace/ tests/
ruff check llmace/ tests/
```

6. **Commit your changes** with clear messages:
```bash
git commit -m "Add feature: brief description"
```

7. **Push to your fork:**
```bash
git push origin feature/your-feature-name
```

8. **Open a Pull Request** with:
   - Clear description of changes
   - Reference to any related issues
   - Screenshots/examples if relevant

## Areas for Contribution

We welcome contributions in these areas:

### High Priority
- Additional LLM provider integrations (Anthropic, Cohere, etc.)
- More comprehensive test coverage
- Performance optimizations
- Documentation improvements

### Medium Priority
- Additional storage backends (Redis, PostgreSQL, etc.)
- More sophisticated deduplication strategies
- Enhanced prompt templates
- Example use cases and tutorials

### Ideas Welcome
- Domain-specific configurations
- Visualization tools for context evolution
- Integration with popular frameworks (LangChain, LlamaIndex)
- CLI tools

## Reporting Issues

When reporting issues, please include:

1. **Description** of the problem
2. **Steps to reproduce**
3. **Expected behavior**
4. **Actual behavior**
5. **Environment details** (Python version, OS, package versions)
6. **Code snippet** demonstrating the issue (if applicable)

Use the issue templates when available.

## Code Review Process

All submissions require review. We'll:

1. Review code for quality and style
2. Ensure tests pass
3. Check documentation is updated
4. Verify the change aligns with project goals

Please be patient - reviews may take a few days.

## Docstring Style

We use Google-style docstrings:

```python
def function_name(arg1: str, arg2: int) -> bool:
    """
    Short description of function.
    
    Longer description if needed, explaining behavior,
    edge cases, and examples.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When validation fails
    
    Example:
        ```python
        result = function_name("test", 42)
        ```
    """
    pass
```

## Questions?

- Open a [Discussion](https://github.com/yourusername/llmace/discussions)
- Join our community channels (if available)
- Email maintainers (if urgent)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to ACE! 🎯

