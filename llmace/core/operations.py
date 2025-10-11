"""Operations for context manipulation: deduplication, pruning, grow-and-refine."""

from typing import Callable, List, Optional
import numpy as np
from llmace.core.context import ACEContext
from llmace.core.schemas import Bullet


class ContextOperations:
    """
    Handles advanced operations on ACE contexts like deduplication and refinement.
    """
    
    def __init__(
        self,
        context: ACEContext,
        embedding_fn: Optional[Callable[[str], List[float]]] = None
    ):
        """
        Initialize context operations.
        
        Args:
            context: The ACEContext to operate on
            embedding_fn: Function that takes text and returns embedding vector.
                         If None, deduplication will be disabled.
        """
        self.context = context
        self.embedding_fn = embedding_fn
        self._embedding_cache: dict[str, np.ndarray] = {}
    
    def deduplicate(self, threshold: Optional[float] = None) -> int:
        """
        Remove duplicate or highly similar bullets using semantic similarity.
        
        Args:
            threshold: Cosine similarity threshold (0-1). Uses config default if None.
        
        Returns:
            Number of bullets removed
        """
        if not self.context.config.enable_deduplication or not self.embedding_fn:
            return 0
        
        threshold = threshold or self.context.config.dedup_threshold
        removed_count = 0
        
        # Process each section independently
        for section in self.context.config.sections:
            bullets = self.context.get_bullets_by_section(section)
            if len(bullets) <= 1:
                continue
            
            # Get embeddings for all bullets in this section
            embeddings = []
            for bullet in bullets:
                embedding = self._get_embedding(bullet.content)
                embeddings.append((bullet, embedding))
            
            # Find and remove duplicates
            to_remove = set()
            for i, (bullet_i, emb_i) in enumerate(embeddings):
                if bullet_i.id in to_remove:
                    continue
                
                for j in range(i + 1, len(embeddings)):
                    bullet_j, emb_j = embeddings[j]
                    if bullet_j.id in to_remove:
                        continue
                    
                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(emb_i, emb_j)
                    
                    if similarity >= threshold:
                        # Keep the bullet with higher score, remove the other
                        if bullet_i.get_score() >= bullet_j.get_score():
                            to_remove.add(bullet_j.id)
                        else:
                            to_remove.add(bullet_i.id)
                            break  # Move to next bullet_i
            
            # Remove duplicates
            for bullet_id in to_remove:
                if self.context.remove_bullet(bullet_id):
                    removed_count += 1
        
        return removed_count
    
    def grow_and_refine(self, threshold: Optional[float] = None) -> dict:
        """
        Apply the grow-and-refine process: deduplicate and prune.
        
        Args:
            threshold: Deduplication threshold (uses config default if None)
        
        Returns:
            Dictionary with statistics about the operation
        """
        stats = {
            "bullets_before": len(self.context),
            "deduped": 0,
            "pruned_negative": 0,
            "pruned_by_limit": 0,
            "bullets_after": 0
        }
        
        # Step 1: Deduplicate
        stats["deduped"] = self.deduplicate(threshold)
        
        # Step 2: Prune negative bullets
        stats["pruned_negative"] = self.context.prune_negative_bullets()
        
        # Step 3: Prune to respect section limits
        stats["pruned_by_limit"] = self.context.prune_by_section_limit()
        
        stats["bullets_after"] = len(self.context)
        
        return stats
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text, with caching.
        
        Args:
            text: Text to embed
        
        Returns:
            Numpy array of embedding
        """
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        
        if not self.embedding_fn:
            raise ValueError("No embedding function provided")
        
        embedding = np.array(self.embedding_fn(text))
        self._embedding_cache[text] = embedding
        return embedding
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
        
        Returns:
            Cosine similarity (0-1)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()

