# Changelog

All notable changes to LLMACE will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-10-15

### Added
- Initial release of LLMACE framework
- Core context management system with bullets and sections
- Automatic reflection using LLM clients
- Curation with delta-based updates
- Semantic deduplication using embeddings
- Grow-and-refine operations with configurable limits
- Manual and automatic modes for reflection
- Serialization (save/load) of evolved contexts
- OpenAI and OpenRouter integration
- LangGraph integration support
- Comprehensive test suite including:
  - Unit tests for core functionality
  - Agentic flow benchmarks
  - FAQ learning benchmarks
  - LangGraph integration examples
- Documentation:
  - Quick start guide
  - Setup guide
  - Testing guide
  - API reference
  - Usage examples

### Features
- **Modular Architecture**: Separate Reflector and Curator components
- **Flexible Client Handling**: Support for separate LLM and embedding clients
- **Type Safety**: Full Pydantic schema validation
- **Configurable Sections**: Customizable context organization
- **Bullet Management**: Add, update, remove, and deduplicate bullets
- **Playbook Generation**: Format context for injection into prompts
- **Retry Logic**: Automatic retry with exponential backoff for LLM calls
- **Structured Outputs**: JSON Schema validation for reliable parsing

### Technical Details
- Python 3.9+ support
- OpenAI SDK 1.0+ compatibility
- Pydantic 2.0+ for data validation
- NumPy for vector operations
- Backoff for retry logic
- Optional dependencies:
  - sentence-transformers for local embeddings
  - langgraph/langchain for agent integration

### Known Limitations
- FAQ benchmarks show modest improvements (context recall vs. reasoning tasks)
- Best suited for multi-step reasoning and agentic workflows
- Requires configured max_bullets_per_section for effective refinement
- Semantic deduplication requires embedding client

## [Unreleased]

### Planned
- Additional benchmark datasets
- Vector database integration for large-scale contexts
- Multi-agent context sharing
- Fine-tuning support for domain-specific reflectors
- Web UI for context visualization and editing
- Additional LLM provider integrations (Anthropic, Google, etc.)
- Performance optimizations for large playbooks
- Context versioning and rollback
- A/B testing framework for comparing contexts

---

For detailed migration guides and breaking changes, see the [documentation](docs/).

