"""Configuration and default templates for ACE."""

from llmace.config.defaults import DEFAULT_SECTIONS, DEFAULT_CONFIG
from llmace.config.templates import (
    REFLECTION_PROMPT_TEMPLATE,
    CURATION_PROMPT_TEMPLATE,
    GENERATOR_INSTRUCTION_TEMPLATE
)

__all__ = [
    "DEFAULT_SECTIONS",
    "DEFAULT_CONFIG",
    "REFLECTION_PROMPT_TEMPLATE",
    "CURATION_PROMPT_TEMPLATE",
    "GENERATOR_INSTRUCTION_TEMPLATE",
]

