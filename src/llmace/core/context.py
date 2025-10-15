"""Core context management for LLMACE."""

import uuid
from typing import Dict, List, Optional
from llmace.core.schemas import Bullet, BulletDelta, ContextConfig


class ACEContext:
    """
    Manages collections of bullets organized by sections.
    
    This is the core data structure that stores and retrieves knowledge
    accumulated through the LLMACE framework.
    """
    
    def __init__(self, config: Optional[ContextConfig] = None):
        """
        Initialize LLMACE context.
        
        Args:
            config: Configuration for context behavior. Uses defaults if None.
        """
        self.config = config or ContextConfig()
        self._bullets: Dict[str, Bullet] = {}  # bullet_id -> Bullet
        self._section_index: Dict[str, List[str]] = {
            section: [] for section in self.config.sections
        }
    
    def add_bullet(
        self,
        section: str,
        content: str,
        bullet_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Bullet:
        """
        Add a new bullet to the context.
        
        Args:
            section: Section to add the bullet to
            content: Content of the bullet
            bullet_id: Optional specific ID (generates UUID if not provided)
            metadata: Optional metadata dict
        
        Returns:
            The created Bullet object
        
        Raises:
            ValueError: If section is not in configured sections
        """
        if section not in self.config.sections:
            raise ValueError(
                f"Section '{section}' not in configured sections: {self.config.sections}"
            )
        
        bullet_id = bullet_id or self._generate_bullet_id(section)
        metadata = metadata or {}
        
        bullet = Bullet(
            id=bullet_id,
            section=section,
            content=content,
            metadata=metadata
        )
        
        self._bullets[bullet_id] = bullet
        if bullet_id not in self._section_index[section]:
            self._section_index[section].append(bullet_id)
        
        return bullet
    
    def get_bullet(self, bullet_id: str) -> Optional[Bullet]:
        """Get a bullet by ID."""
        return self._bullets.get(bullet_id)
    
    def get_bullets_by_section(self, section: str) -> List[Bullet]:
        """Get all bullets in a section."""
        bullet_ids = self._section_index.get(section, [])
        return [self._bullets[bid] for bid in bullet_ids if bid in self._bullets]
    
    def get_all_bullets(self) -> List[Bullet]:
        """Get all bullets across all sections."""
        return list(self._bullets.values())
    
    def update_bullet(
        self,
        bullet_id: str,
        content: Optional[str] = None,
        increment_helpful: bool = False,
        increment_harmful: bool = False,
        metadata: Optional[Dict] = None
    ) -> Optional[Bullet]:
        """
        Update an existing bullet.
        
        Args:
            bullet_id: ID of bullet to update
            content: New content (if provided)
            increment_helpful: Whether to increment helpful counter
            increment_harmful: Whether to increment harmful counter
            metadata: Metadata to merge into existing metadata
        
        Returns:
            Updated Bullet or None if not found
        """
        bullet = self._bullets.get(bullet_id)
        if not bullet:
            return None
        
        if content is not None:
            bullet.content = content.strip()
        
        if increment_helpful:
            bullet.increment_helpful()
        
        if increment_harmful:
            bullet.increment_harmful()
        
        if metadata:
            bullet.metadata.update(metadata)
        
        return bullet
    
    def remove_bullet(self, bullet_id: str) -> bool:
        """
        Remove a bullet from the context.
        
        Args:
            bullet_id: ID of bullet to remove
        
        Returns:
            True if removed, False if not found
        """
        bullet = self._bullets.get(bullet_id)
        if not bullet:
            return False
        
        # Remove from section index
        if bullet.section in self._section_index:
            if bullet_id in self._section_index[bullet.section]:
                self._section_index[bullet.section].remove(bullet_id)
        
        # Remove from bullets dict
        del self._bullets[bullet_id]
        return True
    
    def apply_bullet_tags(self, bullet_tags: List[Dict[str, str]]) -> None:
        """
        Apply tags (helpful/harmful/neutral) to bullets.
        
        Args:
            bullet_tags: List of dicts with 'id' and 'tag' keys
        """
        for tag_info in bullet_tags:
            bullet_id = tag_info.get("id")
            tag = tag_info.get("tag")
            
            if not bullet_id or not tag:
                continue
            
            if tag == "helpful":
                self.update_bullet(bullet_id, increment_helpful=True)
            elif tag == "harmful":
                self.update_bullet(bullet_id, increment_harmful=True)
    
    def merge_delta(self, delta: BulletDelta) -> Bullet:
        """
        Merge a delta update into the context.
        
        Args:
            delta: BulletDelta to apply
        
        Returns:
            The created or updated Bullet
        """
        if delta.operation == "update" and delta.bullet_id:
            # Update existing bullet
            bullet = self.update_bullet(
                bullet_id=delta.bullet_id,
                content=delta.content,
                metadata=delta.metadata
            )
            if bullet:
                return bullet
            # If bullet not found, fall through to add
        
        # Add new bullet
        return self.add_bullet(
            section=delta.section,
            content=delta.content,
            bullet_id=delta.bullet_id,
            metadata=delta.metadata
        )
    
    def merge_deltas(self, deltas: List[BulletDelta]) -> List[Bullet]:
        """
        Merge multiple deltas into the context.
        
        Args:
            deltas: List of BulletDelta objects
        
        Returns:
            List of created/updated bullets
        """
        return [self.merge_delta(delta) for delta in deltas]
    
    def prune_negative_bullets(self) -> int:
        """
        Remove bullets with negative scores (more harmful than helpful).
        
        Returns:
            Number of bullets removed
        """
        if not self.config.prune_negative_bullets:
            return 0
        
        to_remove = [
            bid for bid, bullet in self._bullets.items()
            if bullet.get_score() < 0
        ]
        
        for bullet_id in to_remove:
            self.remove_bullet(bullet_id)
        
        return len(to_remove)
    
    def prune_by_section_limit(self) -> int:
        """
        Prune bullets to respect max_bullets_per_section config.
        Keeps bullets with highest scores.
        
        Returns:
            Number of bullets removed
        """
        if not self.config.max_bullets_per_section:
            return 0
        
        removed_count = 0
        
        for section in self.config.sections:
            bullet_ids = self._section_index[section]
            if len(bullet_ids) <= self.config.max_bullets_per_section:
                continue
            
            # Sort by score (descending)
            bullets = [(bid, self._bullets[bid]) for bid in bullet_ids]
            bullets.sort(key=lambda x: x[1].get_score(), reverse=True)
            
            # Remove lowest scoring bullets
            to_remove = bullets[self.config.max_bullets_per_section:]
            for bullet_id, _ in to_remove:
                self.remove_bullet(bullet_id)
                removed_count += 1
        
        return removed_count
    
    def to_dict(self) -> Dict:
        """
        Serialize context to dictionary.
        
        Returns:
            Dictionary representation of the context
        """
        return {
            "config": self.config.model_dump(),
            "bullets": {
                bid: bullet.model_dump() for bid, bullet in self._bullets.items()
            },
            "section_index": self._section_index
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ACEContext":
        """
        Deserialize context from dictionary.
        
        Args:
            data: Dictionary with context data
        
        Returns:
            Reconstructed ACEContext
        """
        config = ContextConfig(**data["config"])
        context = cls(config=config)
        
        # Restore bullets
        for bullet_id, bullet_data in data["bullets"].items():
            bullet = Bullet(**bullet_data)
            context._bullets[bullet_id] = bullet
        
        # Restore section index
        context._section_index = data["section_index"]
        
        return context
    
    def _generate_bullet_id(self, section: str) -> str:
        """Generate a unique bullet ID with section prefix."""
        prefix = section[:3].lower()
        return f"{prefix}-{uuid.uuid4().hex[:8]}"
    
    def __len__(self) -> int:
        """Return the number of bullets in the context."""
        return len(self._bullets)
    
    def __repr__(self) -> str:
        """String representation of the context."""
        return f"ACEContext(bullets={len(self._bullets)}, sections={len(self.config.sections)})"

