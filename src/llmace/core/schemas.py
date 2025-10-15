"""Pydantic schemas for LLMLLMACE data structures and validation."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class Bullet(BaseModel):
    """
    A single unit of knowledge in the LLMACE context.
    
    Bullets are atomic pieces of information like strategies, insights,
    common mistakes, or domain concepts that accumulate over time.
    """
    
    id: str = Field(..., description="Unique identifier for this bullet")
    section: str = Field(..., description="Section this bullet belongs to (e.g., 'strategies', 'insights')")
    content: str = Field(..., description="The actual content/text of this bullet")
    helpful_count: int = Field(default=0, ge=0, description="Number of times marked as helpful")
    harmful_count: int = Field(default=0, ge=0, description="Number of times marked as harmful")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        """Ensure content is not empty."""
        if not v or not v.strip():
            raise ValueError("Bullet content cannot be empty")
        return v.strip()
    
    def increment_helpful(self) -> None:
        """Increment the helpful counter."""
        self.helpful_count += 1
    
    def increment_harmful(self) -> None:
        """Increment the harmful counter."""
        self.harmful_count += 1
    
    def get_score(self) -> int:
        """Get the net score (helpful - harmful)."""
        return self.helpful_count - self.harmful_count
    
    def format_with_metadata(self) -> str:
        """Format bullet with metadata for display."""
        return f"[{self.id}] helpful={self.helpful_count} harmful={self.harmful_count} :: {self.content}"


class BulletDelta(BaseModel):
    """
    Represents a proposed change to the context (add new or update existing bullet).
    """
    
    operation: str = Field(..., description="Operation type: 'add' or 'update'")
    bullet_id: Optional[str] = Field(None, description="ID of existing bullet (for updates)")
    section: str = Field(..., description="Section for this bullet")
    content: str = Field(..., description="Content of the bullet")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @field_validator("operation")
    @classmethod
    def validate_operation(cls, v: str) -> str:
        """Ensure operation is valid."""
        if v not in ["add", "update"]:
            raise ValueError("Operation must be 'add' or 'update'")
        return v
    
    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        """Ensure content is not empty."""
        if not v or not v.strip():
            raise ValueError("Delta content cannot be empty")
        return v.strip()


class ContextConfig(BaseModel):
    """
    Configuration for LLMLLMACE context behavior.
    """
    
    sections: List[str] = Field(
        default=["strategies", "insights", "common_mistakes", "best_practices", "patterns"],
        description="Sections to organize bullets into"
    )
    dedup_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for deduplication (0-1)"
    )
    max_bullets_per_section: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum bullets per section (None for unlimited)"
    )
    enable_deduplication: bool = Field(
        default=True,
        description="Whether to enable semantic deduplication"
    )
    prune_negative_bullets: bool = Field(
        default=False,
        description="Whether to remove bullets with negative scores"
    )
    
    @field_validator("sections")
    @classmethod
    def sections_not_empty(cls, v: List[str]) -> List[str]:
        """Ensure at least one section is defined."""
        if not v:
            raise ValueError("At least one section must be defined")
        return v


class ReflectionInput(BaseModel):
    """
    Input for reflection process.
    """
    
    query: str = Field(..., description="The original query/task")
    response: str = Field(..., description="The generated response")
    success: bool = Field(..., description="Whether the task was successful")
    feedback: Optional[str] = Field(None, description="Optional feedback about the execution")
    ground_truth: Optional[str] = Field(None, description="Optional ground truth for comparison")
    execution_trace: Optional[str] = Field(None, description="Optional execution trace/logs")
    used_bullet_ids: List[str] = Field(
        default_factory=list,
        description="IDs of bullets that were used/retrieved for this task"
    )


class ReflectionOutput(BaseModel):
    """
    Output from reflection process.
    """
    
    reasoning: str = Field(..., description="Chain of thought about what happened")
    error_identification: Optional[str] = Field(
        None,
        description="What specifically went wrong (if applicable)"
    )
    root_cause_analysis: Optional[str] = Field(
        None,
        description="Why the error occurred (if applicable)"
    )
    correct_approach: Optional[str] = Field(
        None,
        description="What should have been done instead"
    )
    key_insight: str = Field(..., description="Main insight or lesson learned")
    bullet_tags: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Tags for bullets that were used: [{'id': 'bullet-001', 'tag': 'helpful'}]"
    )


class CurationOutput(BaseModel):
    """
    Output from curation process - delta updates to apply to context.
    """
    
    reasoning: str = Field(..., description="Reasoning behind the proposed changes")
    deltas: List[BulletDelta] = Field(
        default_factory=list,
        description="List of delta operations to apply"
    )

