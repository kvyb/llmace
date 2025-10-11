"""Utilities for working with embeddings."""

from typing import Callable, List, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def create_embedding_function(
    client: OpenAI,
    model: str = "text-embedding-3-small"
) -> Callable[[str], List[float]]:
    """
    Create an embedding function from an OpenAI client.
    
    Args:
        client: OpenAI client instance
        model: Embedding model to use
    
    Returns:
        Function that takes text and returns embedding vector
    
    Example:
        ```python
        from openai import OpenAI
        from llmace.utils.embeddings import create_embedding_function
        
        client = OpenAI(api_key="sk-...")
        embedding_fn = create_embedding_function(client)
        
        # Use with ACE
        ace = ACE(embedding_fn=embedding_fn)
        ```
    """
    def embed(text: str) -> List[float]:
        """Generate embedding for text."""
        response = client.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding
    
    return embed


def get_default_embedding_model(provider: Optional[str] = None) -> str:
    """
    Get default embedding model for a provider.
    
    Args:
        provider: Provider name (e.g., "openai", "azure")
    
    Returns:
        Default model name
    """
    # OpenAI and most compatible providers
    return "text-embedding-3-small"

