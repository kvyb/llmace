"""
LLMACE (Agentic Context Engineering) - A framework for evolving contexts in LLM workflows.

This package provides tools for building comprehensive, evolving playbooks that accumulate
and organize strategies through modular generation, reflection, and curation processes.
"""

from llmace.ace import LLMACE
from llmace.core.context import Bullet, ACEContext
from llmace.core.schemas import BulletDelta, ContextConfig
from llmace.utils.embeddings import create_embedding_function

__version__ = "0.1.0"

__all__ = [
    "LLMACE",
    "Bullet",
    "ACEContext",
    "BulletDelta",
    "ContextConfig",
    "create_embedding_function",
]

