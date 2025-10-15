"""Reflector for analyzing task executions and extracting insights."""

import json
from typing import Optional
import backoff

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from llmace.core.schemas import ReflectionInput, ReflectionOutput
from llmace.config.templates import REFLECTION_PROMPT_TEMPLATE


class Reflector:
    """
    Analyzes task executions to extract insights and lessons learned.
    
    Supports both automatic mode (calls LLM) and manual mode (accepts pre-computed results).
    """
    
    def __init__(
        self,
        llm_client: Optional[OpenAI] = None,
        prompt_template: Optional[str] = None
    ):
        """
        Initialize Reflector.
        
        Args:
            llm_client: Optional LLM client (OpenAI-compatible) for automatic reflection
            prompt_template: Custom reflection prompt template (uses default if None)
        """
        self.llm_client = llm_client
        self.prompt_template = prompt_template or REFLECTION_PROMPT_TEMPLATE
    
    def reflect(
        self,
        reflection_input: ReflectionInput,
        reflection_result: Optional[dict] = None
    ) -> ReflectionOutput:
        """
        Perform reflection on a task execution.
        
        Args:
            reflection_input: Input data for reflection
            reflection_result: Pre-computed reflection result (for manual mode).
                             If provided, LLM will not be called.
        
        Returns:
            ReflectionOutput with insights and analysis
        
        Raises:
            ValueError: If neither llm_client nor reflection_result is provided
        """
        if reflection_result is not None:
            # Manual mode: use provided result
            return ReflectionOutput(**reflection_result)
        
        if self.llm_client is None:
            raise ValueError(
                "Either llm_client must be provided during initialization "
                "or reflection_result must be passed to reflect()"
            )
        
        # Automatic mode: call LLM
        return self._reflect_with_llm(reflection_input)
    
    def _reflect_with_llm(self, reflection_input: ReflectionInput) -> ReflectionOutput:
        """
        Perform reflection using LLM.
        
        Args:
            reflection_input: Input data for reflection
        
        Returns:
            ReflectionOutput from LLM
        """
        # Build the prompt
        prompt = self._build_reflection_prompt(reflection_input)
        
        # Define JSON schema matching ReflectionOutput model
        json_schema = {
            "name": "reflection_output",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string"},
                    "error_identification": {"type": ["string", "null"]},
                    "root_cause_analysis": {"type": ["string", "null"]},
                    "correct_approach": {"type": ["string", "null"]},
                    "key_insight": {"type": "string"},
                    "bullet_tags": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "tag": {"type": "string"}
                            },
                            "required": ["id", "tag"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["reasoning", "key_insight"],
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
                    {"role": "system", "content": "You are an expert analyst. Respond with valid JSON only."},
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
        
        # Apply bullet tags if provided in input
        if reflection_input.used_bullet_ids:
            # For now, we don't auto-tag bullets during reflection
            # This would require additional logic to determine helpful/harmful
            pass
        
        return ReflectionOutput(**result)
    
    def _build_reflection_prompt(self, reflection_input: ReflectionInput) -> str:
        """
        Build the reflection prompt from input.
        
        Args:
            reflection_input: Input data
        
        Returns:
            Formatted prompt string
        """
        # Build optional sections
        feedback_section = ""
        if reflection_input.feedback:
            feedback_section = f"Feedback: {reflection_input.feedback}"
        
        ground_truth_section = ""
        if reflection_input.ground_truth:
            ground_truth_section = f"Ground Truth: {reflection_input.ground_truth}"
        
        execution_trace_section = ""
        if reflection_input.execution_trace:
            execution_trace_section = f"Execution Trace:\n{reflection_input.execution_trace}"
        
        return self.prompt_template.format(
            query=reflection_input.query,
            response=reflection_input.response,
            success=reflection_input.success,
            feedback_section=feedback_section,
            ground_truth_section=ground_truth_section,
            execution_trace_section=execution_trace_section
        )

