"""Default configuration values for ACE."""

from typing import Dict, Any

# Default sections for universal multi-turn agentic workflows
DEFAULT_SECTIONS = [
    "strategies",          # High-level strategies and approaches
    "insights",           # Key insights and lessons learned
    "common_mistakes",    # Pitfalls and errors to avoid
    "best_practices",     # Proven best practices
    "patterns",           # Recurring patterns and solutions
]

# Default configuration
DEFAULT_CONFIG: Dict[str, Any] = {
    "sections": DEFAULT_SECTIONS,
    "dedup_threshold": 0.85,
    "max_bullets_per_section": None,
    "enable_deduplication": True,
    "prune_negative_bullets": False,
}

