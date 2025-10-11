"""OpenAI integration utilities for ACE."""

from typing import List, Dict, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from llmace.core.context import ACEContext
from llmace.utils.formatting import format_playbook_for_prompt
from llmace.config.templates import GENERATOR_INSTRUCTION_TEMPLATE


def inject_playbook_into_messages(
    messages: List[Dict[str, str]],
    context: ACEContext,
    position: str = "system",
    include_metadata: bool = False,
    min_score: int = 0,
    instruction_template: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Inject ACE playbook into OpenAI messages array.
    
    Args:
        messages: List of message dicts (OpenAI format)
        context: ACEContext to inject
        position: Where to inject - "system" (prepend to system message),
                 "before_user" (add before first user message),
                 "after_system" (add as separate message after system)
        include_metadata: Whether to include bullet metadata
        min_score: Minimum bullet score to include
        instruction_template: Custom instruction template
    
    Returns:
        New messages list with playbook injected
    """
    # Format playbook
    playbook = format_playbook_for_prompt(
        context,
        include_metadata=include_metadata,
        min_score=min_score
    )
    
    if not playbook:
        return messages  # No playbook to inject
    
    # Format with instructions
    template = instruction_template or GENERATOR_INSTRUCTION_TEMPLATE
    playbook_content = template.format(playbook=playbook)
    
    # Copy messages
    new_messages = messages.copy()
    
    if position == "system":
        # Prepend to existing system message or create one
        system_msg_idx = None
        for i, msg in enumerate(new_messages):
            if msg.get("role") == "system":
                system_msg_idx = i
                break
        
        if system_msg_idx is not None:
            # Prepend to existing system message
            existing_content = new_messages[system_msg_idx]["content"]
            new_messages[system_msg_idx]["content"] = f"{playbook_content}\n\n{existing_content}"
        else:
            # Create new system message at start
            new_messages.insert(0, {"role": "system", "content": playbook_content})
    
    elif position == "after_system":
        # Add as separate message after system message
        insert_idx = 0
        for i, msg in enumerate(new_messages):
            if msg.get("role") == "system":
                insert_idx = i + 1
                break
        new_messages.insert(insert_idx, {"role": "system", "content": playbook_content})
    
    elif position == "before_user":
        # Add before first user message
        insert_idx = 0
        for i, msg in enumerate(new_messages):
            if msg.get("role") == "user":
                insert_idx = i
                break
        new_messages.insert(insert_idx, {"role": "system", "content": playbook_content})
    
    else:
        raise ValueError(f"Invalid position: {position}")
    
    return new_messages


class ACEOpenAIWrapper:
    """
    Wrapper around OpenAI client that automatically injects ACE playbook.
    
    This provides a convenient way to use ACE with OpenAI-style APIs.
    """
    
    def __init__(
        self,
        client: OpenAI,
        context: ACEContext,
        auto_inject: bool = True,
        injection_position: str = "system",
        include_metadata: bool = False,
        min_score: int = 0
    ):
        """
        Initialize ACE OpenAI wrapper.
        
        Args:
            client: OpenAI client (or compatible)
            context: ACEContext to use
            auto_inject: Whether to automatically inject playbook
            injection_position: Where to inject playbook in messages
            include_metadata: Whether to include bullet metadata
            min_score: Minimum bullet score to include
        """
        self.client = client
        self.context = context
        self.auto_inject = auto_inject
        self.injection_position = injection_position
        self.include_metadata = include_metadata
        self.min_score = min_score
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        inject_playbook: Optional[bool] = None,
        **kwargs
    ) -> Any:
        """
        Create chat completion with optional playbook injection.
        
        Args:
            messages: Messages array
            inject_playbook: Override auto_inject setting
            **kwargs: Additional arguments for OpenAI API
        
        Returns:
            OpenAI completion response
        """
        should_inject = inject_playbook if inject_playbook is not None else self.auto_inject
        
        if should_inject and len(self.context) > 0:
            messages = inject_playbook_into_messages(
                messages=messages,
                context=self.context,
                position=self.injection_position,
                include_metadata=self.include_metadata,
                min_score=self.min_score
            )
        
        return self.client.chat.completions.create(messages=messages, **kwargs)
    
    def update_context(self, new_context: ACEContext) -> None:
        """Update the context being used."""
        self.context = new_context

