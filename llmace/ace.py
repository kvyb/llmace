"""Main ACE class - high-level interface for Agentic Context Engineering."""

import json
from typing import Any, Callable, Dict, List, Optional
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # Type will be Optional[OpenAI] which allows None

from llmace.core.context import ACEContext
from llmace.core.schemas import (
    ContextConfig,
    ReflectionInput,
    ReflectionOutput,
    CurationOutput
)
from llmace.core.operations import ContextOperations
from llmace.reflection.reflector import Reflector
from llmace.reflection.curator import Curator
from llmace.utils.formatting import format_playbook_for_prompt
from llmace.utils.logging import setup_logger


class ACE:
    """
    Main ACE (Agentic Context Engineering) class.
    
    This provides a high-level interface for building and evolving contexts
    through reflection and curation.
    
    Example:
        ```python
        from llmace import ACE
        from openai import OpenAI
        
        # Initialize with OpenAI client
        client = OpenAI(api_key="...")
        ace = ACE(llm_client=client)
        
        # Get playbook for prompt injection
        playbook = ace.get_playbook()
        
        # Reflect on an execution
        ace.reflect(
            query="Solve this problem...",
            response="Here's my solution...",
            success=True
        )
        
        # Save/load context
        ace.save("my_context.json")
        ace = ACE.load("my_context.json", llm_client=client)
        ```
    """
    
    def __init__(
        self,
        llm_client: Optional[OpenAI] = None,
        embedding_fn: Optional[Callable[[str], List[float]]] = None,
        embedding_client: Optional[OpenAI] = None,
        embedding_model: str = "text-embedding-3-small",
        config: Optional[ContextConfig] = None,
        context: Optional[ACEContext] = None,
        reflection_prompt: Optional[str] = None,
        curation_prompt: Optional[str] = None,
        enable_logging: bool = False
    ):
        """
        Initialize ACE.
        
        Args:
            llm_client: OpenAI-compatible client for auto-reflection (optional)
            embedding_fn: Function to generate embeddings for deduplication (optional)
            embedding_client: OpenAI client for embeddings (alternative to embedding_fn)
            embedding_model: Model to use with embedding_client (default: text-embedding-3-small)
            config: Context configuration (uses defaults if None)
            context: Existing ACEContext to use (creates new if None)
            reflection_prompt: Custom reflection prompt template
            curation_prompt: Custom curation prompt template
            enable_logging: Whether to enable logging
        
        Example:
            ```python
            from openai import OpenAI
            from llmace import ACE
            
            # LLM client (could be OpenRouter)
            llm_client = OpenAI(
                api_key="sk-or-...",
                base_url="https://openrouter.ai/api/v1"
            )
            
            # Separate OpenAI client for embeddings
            embedding_client = OpenAI(api_key="sk-...")
            
            ace = ACE(
                llm_client=llm_client,
                embedding_client=embedding_client
            )
            ```
        """
        # Setup logging
        self.logger = setup_logger() if enable_logging else None
        
        # Handle embedding_client parameter
        if embedding_client is not None and embedding_fn is None:
            from llmace.utils.embeddings import create_embedding_function
            
            if self.logger:
                self.logger.info(f"Creating embedding function with model: {embedding_model}")
            
            embedding_fn = create_embedding_function(
                client=embedding_client,
                model=embedding_model
            )
        
        # Initialize context
        self.context = context or ACEContext(config=config)
        
        # Initialize operations
        self.operations = ContextOperations(
            context=self.context,
            embedding_fn=embedding_fn
        )
        
        # Initialize reflection components
        self.reflector = Reflector(
            llm_client=llm_client,
            prompt_template=reflection_prompt
        )
        
        self.curator = Curator(
            llm_client=llm_client,
            prompt_template=curation_prompt
        )
        
        self.llm_client = llm_client
        self.embedding_fn = embedding_fn
    
    def get_playbook(
        self,
        include_metadata: bool = False,
        min_score: int = 0,
        sections: Optional[List[str]] = None
    ) -> str:
        """
        Get formatted playbook string for prompt injection.
        
        Args:
            include_metadata: Whether to include bullet metadata
            min_score: Minimum bullet score to include
            sections: Specific sections to include (all if None)
        
        Returns:
            Formatted playbook string
        """
        from llmace.utils.formatting import PlaybookFormatter
        
        formatter = PlaybookFormatter(include_metadata=include_metadata)
        return formatter.format_playbook(
            context=self.context,
            sections=sections,
            min_score=min_score
        )
    
    def reflect(
        self,
        query: str,
        response: str,
        success: bool,
        feedback: Optional[str] = None,
        ground_truth: Optional[str] = None,
        execution_trace: Optional[str] = None,
        used_bullet_ids: Optional[List[str]] = None,
        reflection_result: Optional[dict] = None,
        curation_result: Optional[dict] = None,
        task_context: str = "",
        auto_update: bool = True,
        run_grow_and_refine: bool = True
    ) -> Dict[str, Any]:
        """
        Reflect on a task execution and optionally update context.
        
        This is the main method for evolving the context based on feedback.
        
        Args:
            query: The original query/task
            response: The generated response
            success: Whether the task was successful
            feedback: Optional feedback about execution
            ground_truth: Optional ground truth for comparison
            execution_trace: Optional execution trace/logs
            used_bullet_ids: IDs of bullets that were used
            reflection_result: Pre-computed reflection (manual mode)
            curation_result: Pre-computed curation (manual mode)
            task_context: Context about the task for curator
            auto_update: Whether to automatically apply updates to context
            run_grow_and_refine: Whether to run grow-and-refine after update
        
        Returns:
            Dictionary with reflection, curation, and update statistics
        """
        # Step 1: Reflect
        reflection_input = ReflectionInput(
            query=query,
            response=response,
            success=success,
            feedback=feedback,
            ground_truth=ground_truth,
            execution_trace=execution_trace,
            used_bullet_ids=used_bullet_ids or []
        )
        
        reflection = self.reflector.reflect(
            reflection_input=reflection_input,
            reflection_result=reflection_result
        )
        
        if self.logger:
            self.logger.info(f"Reflection complete. Key insight: {reflection.key_insight[:100]}...")
        
        # Step 2: Apply bullet tags if any
        if reflection.bullet_tags:
            self.context.apply_bullet_tags(reflection.bullet_tags)
        
        # Step 3: Curate (extract insights and generate deltas)
        curation = self.curator.curate(
            reflection=reflection,
            context=self.context,
            task_context=task_context,
            curation_result=curation_result
        )
        
        if self.logger:
            self.logger.info(f"Curation complete. Generated {len(curation.deltas)} deltas")
        
        # Step 4: Apply updates if auto_update
        update_stats = {}
        if auto_update and curation.deltas:
            bullets_before = len(self.context)
            self.context.merge_deltas(curation.deltas)
            bullets_after = len(self.context)
            
            # Run grow-and-refine if enabled
            if run_grow_and_refine:
                refine_stats = self.operations.grow_and_refine()
                update_stats = refine_stats
            else:
                update_stats = {
                    "bullets_before": bullets_before,
                    "bullets_after": bullets_after,
                    "added": bullets_after - bullets_before
                }
            
            if self.logger:
                self.logger.info(f"Context updated: {update_stats}")
        
        return {
            "reflection": reflection.model_dump(),
            "curation": curation.model_dump(),
            "update_stats": update_stats
        }
    
    def add_bullet(
        self,
        section: str,
        content: str,
        bullet_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Manually add a bullet to the context.
        
        Args:
            section: Section to add bullet to
            content: Content of the bullet
            bullet_id: Optional specific ID
            metadata: Optional metadata
        
        Returns:
            ID of the created bullet
        """
        bullet = self.context.add_bullet(
            section=section,
            content=content,
            bullet_id=bullet_id,
            metadata=metadata
        )
        return bullet.id
    
    def grow_and_refine(self, threshold: Optional[float] = None) -> Dict[str, int]:
        """
        Run grow-and-refine process manually.
        
        Args:
            threshold: Deduplication threshold
        
        Returns:
            Statistics dictionary
        """
        return self.operations.grow_and_refine(threshold=threshold)
    
    def to_dict(self) -> Dict:
        """
        Serialize ACE instance to dictionary.
        
        Returns:
            Dictionary representation
        """
        return self.context.to_dict()
    
    @classmethod
    def from_dict(
        cls,
        data: Dict,
        llm_client: Optional[OpenAI] = None,
        embedding_fn: Optional[Callable[[str], List[float]]] = None,
        embedding_client: Optional[OpenAI] = None,
        **kwargs
    ) -> "ACE":
        """
        Deserialize ACE instance from dictionary.
        
        Args:
            data: Dictionary with context data
            llm_client: LLM client for reflection
            embedding_fn: Embedding function
            embedding_client: OpenAI client for embeddings
            **kwargs: Additional arguments for ACE constructor
        
        Returns:
            Reconstructed ACE instance
        """
        context = ACEContext.from_dict(data)
        return cls(
            llm_client=llm_client,
            embedding_fn=embedding_fn,
            embedding_client=embedding_client,
            context=context,
            **kwargs
        )
    
    def save(self, filepath: str) -> None:
        """
        Save context to JSON file.
        
        Args:
            filepath: Path to save file
        """
        data = self.to_dict()
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        if self.logger:
            self.logger.info(f"Context saved to {filepath}")
    
    @classmethod
    def load(
        cls,
        filepath: str,
        llm_client: Optional[OpenAI] = None,
        embedding_fn: Optional[Callable[[str], List[float]]] = None,
        embedding_client: Optional[OpenAI] = None,
        **kwargs
    ) -> "ACE":
        """
        Load context from JSON file.
        
        Args:
            filepath: Path to load file
            llm_client: LLM client for reflection
            embedding_fn: Embedding function
            embedding_client: OpenAI client for embeddings
            **kwargs: Additional arguments for ACE constructor
        
        Returns:
            Loaded ACE instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(
            data=data,
            llm_client=llm_client,
            embedding_fn=embedding_fn,
            embedding_client=embedding_client,
            **kwargs
        )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"ACE(bullets={len(self.context)}, sections={len(self.context.config.sections)})"

