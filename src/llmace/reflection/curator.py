"""Curator for extracting insights from reflections and generating context updates."""

import json
from typing import Optional, List
import backoff

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from llmace.core.schemas import ReflectionOutput, CurationOutput, BulletDelta
from llmace.core.context import ACEContext
from llmace.config.templates import CURATION_PROMPT_TEMPLATE
from llmace.utils.formatting import format_playbook_for_prompt


class Curator:
    """
    Extracts actionable insights from reflections and generates delta updates.
    
    Supports both automatic mode (calls LLM) and manual mode (accepts pre-computed results).
    """
    
    def __init__(
        self,
        llm_client: Optional[OpenAI] = None,
        prompt_template: Optional[str] = None
    ):
        """
        Initialize Curator.
        
        Args:
            llm_client: Optional LLM client (OpenAI-compatible) for automatic curation
            prompt_template: Custom curation prompt template (uses default if None)
        """
        self.llm_client = llm_client
        self.prompt_template = prompt_template or CURATION_PROMPT_TEMPLATE
    
    def curate(
        self,
        reflection: ReflectionOutput,
        context: ACEContext,
        task_context: str = "",
        curation_result: Optional[dict] = None
    ) -> CurationOutput:
        """
        Curate insights from reflection and generate delta updates.
        
        Args:
            reflection: ReflectionOutput from reflection process
            context: Current ACEContext (to avoid redundant additions)
            task_context: Optional context about the task
            curation_result: Pre-computed curation result (for manual mode).
                           If provided, LLM will not be called.
        
        Returns:
            CurationOutput with reasoning and delta updates
        
        Raises:
            ValueError: If neither llm_client nor curation_result is provided
        """
        if curation_result is not None:
            # Manual mode: use provided result
            return self._parse_curation_result(curation_result)
        
        if self.llm_client is None:
            raise ValueError(
                "Either llm_client must be provided during initialization "
                "or curation_result must be passed to curate()"
            )
        
        # Automatic mode: call LLM
        return self._curate_with_llm(reflection, context, task_context)
    
    def _curate_with_llm(
        self,
        reflection: ReflectionOutput,
        context: ACEContext,
        task_context: str
    ) -> CurationOutput:
        """
        Perform curation using LLM.
        
        Args:
            reflection: ReflectionOutput from reflection
            context: Current ACEContext
            task_context: Context about the task
        
        Returns:
            CurationOutput from LLM
        """
        # Build the prompt
        prompt = self._build_curation_prompt(reflection, context, task_context)
        
        # Define JSON schema for structured output
        json_schema = {
            "name": "curation_output",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string"},
                    "deltas": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "operation": {
                                    "type": "string",
                                    "enum": ["add", "update"]
                                },
                                "bullet_id": {
                                    "type": ["string", "null"]
                                },
                                "section": {"type": "string"},
                                "content": {"type": "string"},
                                "metadata": {"type": "object", "additionalProperties": True}
                            },
                            "required": ["operation", "section", "content"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["reasoning", "deltas"],
                "additionalProperties": False
            }
        }
        
        # Call LLM with structured output and retry on failures
        @backoff.on_exception(
            backoff.expo,
            (json.JSONDecodeError, Exception),
            max_tries=3,
            max_time=30
        )
        def call_llm_with_retry():
            response = self.llm_client.chat.completions.create(
                model=getattr(self.llm_client, 'default_model', 'google/gemini-2.5-flash'),
                messages=[
                    {"role": "system", "content": "You are a knowledge curator. Respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                response_format={
                    "type": "json_schema",
                    "json_schema": json_schema
                }
            )
            content = response.choices[0].message.content
            return json.loads(content)
        
        result = call_llm_with_retry()
        
        return self._parse_curation_result(result)
    
    def _build_curation_prompt(
        self,
        reflection: ReflectionOutput,
        context: ACEContext,
        task_context: str
    ) -> str:
        """
        Build the curation prompt.
        
        Args:
            reflection: ReflectionOutput
            context: Current ACEContext
            task_context: Task context string
        
        Returns:
            Formatted prompt string
        """
        # Format current playbook
        current_playbook = format_playbook_for_prompt(context, include_metadata=False)
        if not current_playbook:
            current_playbook = "[Empty playbook - no bullets yet]"
        
        # Format reflection
        reflection_text = f"""
Reasoning: {reflection.reasoning}
Key Insight: {reflection.key_insight}
"""
        if reflection.error_identification:
            reflection_text += f"Error: {reflection.error_identification}\n"
        if reflection.root_cause_analysis:
            reflection_text += f"Root Cause: {reflection.root_cause_analysis}\n"
        if reflection.correct_approach:
            reflection_text += f"Correct Approach: {reflection.correct_approach}\n"
        
        return self.prompt_template.format(
            current_playbook=current_playbook,
            reflection=reflection_text.strip(),
            task_context=task_context or "[No specific task context provided]",
            sections=", ".join(context.config.sections)
        )
    
    def _parse_curation_result(self, result: dict) -> CurationOutput:
        """
        Parse curation result into CurationOutput.
        
        Args:
            result: Dictionary with curation results
        
        Returns:
            CurationOutput object
        """
        # Parse deltas
        deltas = []
        for delta_data in result.get("deltas", []):
            deltas.append(BulletDelta(**delta_data))
        
        return CurationOutput(
            reasoning=result.get("reasoning", ""),
            deltas=deltas
        )

