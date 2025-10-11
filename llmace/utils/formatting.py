"""Formatting utilities for generating playbooks."""

from typing import List, Optional
from llmace.core.context import ACEContext
from llmace.core.schemas import Bullet


class PlaybookFormatter:
    """
    Formats ACE context into structured playbook strings for prompt injection.
    """
    
    def __init__(
        self,
        include_metadata: bool = True,
        section_header_template: str = "**{section_name}**",
        bullet_template: str = "- {content}",
        bullet_with_meta_template: str = "- [{id}] helpful={helpful} harmful={harmful} :: {content}"
    ):
        """
        Initialize playbook formatter.
        
        Args:
            include_metadata: Whether to include bullet metadata in output
            section_header_template: Template for section headers
            bullet_template: Template for bullets without metadata
            bullet_with_meta_template: Template for bullets with metadata
        """
        self.include_metadata = include_metadata
        self.section_header_template = section_header_template
        self.bullet_template = bullet_template
        self.bullet_with_meta_template = bullet_with_meta_template
    
    def format_playbook(
        self,
        context: ACEContext,
        sections: Optional[List[str]] = None,
        min_score: Optional[int] = None
    ) -> str:
        """
        Format context as a playbook string.
        
        Args:
            context: ACEContext to format
            sections: Specific sections to include (all if None)
            min_score: Minimum bullet score to include (no filter if None)
        
        Returns:
            Formatted playbook string
        """
        sections_to_use = sections or context.config.sections
        output_lines = []
        
        for section in sections_to_use:
            bullets = context.get_bullets_by_section(section)
            
            # Filter by score if specified
            if min_score is not None:
                bullets = [b for b in bullets if b.get_score() >= min_score]
            
            if not bullets:
                continue
            
            # Add section header
            section_name = self._format_section_name(section)
            output_lines.append(self.section_header_template.format(section_name=section_name))
            output_lines.append("")
            
            # Add bullets
            for bullet in bullets:
                formatted_bullet = self._format_bullet(bullet)
                output_lines.append(formatted_bullet)
            
            output_lines.append("")  # Empty line between sections
        
        return "\n".join(output_lines).strip()
    
    def format_section(self, context: ACEContext, section: str) -> str:
        """
        Format a single section.
        
        Args:
            context: ACEContext to format
            section: Section name
        
        Returns:
            Formatted section string
        """
        return self.format_playbook(context, sections=[section])
    
    def _format_bullet(self, bullet: Bullet) -> str:
        """Format a single bullet."""
        if self.include_metadata:
            return self.bullet_with_meta_template.format(
                id=bullet.id,
                helpful=bullet.helpful_count,
                harmful=bullet.harmful_count,
                content=bullet.content
            )
        else:
            return self.bullet_template.format(content=bullet.content)
    
    def _format_section_name(self, section: str) -> str:
        """Format section name for display (capitalize and replace underscores)."""
        return section.replace("_", " ").title()


def format_playbook_for_prompt(
    context: ACEContext,
    include_metadata: bool = False,
    min_score: int = 0
) -> str:
    """
    Convenience function to format playbook for prompt injection.
    
    Args:
        context: ACEContext to format
        include_metadata: Whether to include bullet metadata
        min_score: Minimum score for bullets to include
    
    Returns:
        Formatted playbook string ready for prompts
    """
    formatter = PlaybookFormatter(include_metadata=include_metadata)
    return formatter.format_playbook(context, min_score=min_score)

