"""Prompt templates for ACE reflection and curation."""

# Reflection prompt - analyzes what went right/wrong in a task execution
REFLECTION_PROMPT_TEMPLATE = """You are an expert analyst tasked with reflecting on an AI agent's performance.

**Task Context:**
Query: {query}

**Generated Response:**
{response}

**Execution Details:**
Success: {success}
{feedback_section}
{ground_truth_section}
{execution_trace_section}

**Instructions:**
- Carefully analyze what happened during this execution
- If the task failed, identify what went wrong and why
- If the task succeeded, identify what strategies worked well
- Consider both conceptual errors and execution mistakes
- Provide actionable insights that could help prevent future mistakes or replicate success
- Focus on generalizable lessons, not just this specific case

**Output Format:**
Respond with a JSON object containing:
{{
    "reasoning": "Your detailed chain of thought analyzing what happened",
    "error_identification": "What specifically went wrong (if applicable, null otherwise)",
    "root_cause_analysis": "Why the error occurred (if applicable, null otherwise)",
    "correct_approach": "What should have been done instead (if applicable, null otherwise)",
    "key_insight": "The main insight or lesson learned from this execution",
    "bullet_tags": []
}}

Note: bullet_tags should remain empty for now - it will be populated later if specific playbook bullets were used.
"""

# Curation prompt - extracts actionable insights and creates delta updates
CURATION_PROMPT_TEMPLATE = """You are a knowledge curator. Your job is to extract actionable insights from a reflection and propose updates to an existing playbook.

**Current Playbook:**
{current_playbook}

**Recent Reflection:**
{reflection}

**Task Context:**
{task_context}

**Instructions:**
- Review the reflection and identify NEW insights that are MISSING from the current playbook
- Avoid redundancy - only propose additions if the insight is not already covered
- Focus on generalizable, reusable knowledge
- Each insight should be clear, concise, and actionable
- Organize insights by section: {sections}

**Output Format:**
Respond with a JSON object containing:
{{
    "reasoning": "Your analysis of what new knowledge should be added and why",
    "deltas": [
        {{
            "operation": "add",
            "bullet_id": null,
            "section": "strategies",
            "content": "New strategy to add...",
            "metadata": {{}}
        }}
    ]
}}

Guidelines for operations:
- "add": Create a new bullet (bullet_id should be null)
- "update": Update existing bullet (provide bullet_id)

If no new insights are needed, return an empty deltas list.
"""

# Generator instruction template - how to use the playbook
GENERATOR_INSTRUCTION_TEMPLATE = """**ACE Playbook Available:**

You have access to a curated playbook containing strategies, insights, and best practices accumulated from previous executions. Use this playbook to inform your approach.

**Playbook Contents:**
{playbook}

**Instructions for using the playbook:**
1. Review the playbook sections relevant to your current task
2. Apply strategies and best practices where appropriate
3. Avoid common mistakes listed in the playbook
4. Use insights to guide your reasoning and decision-making
5. The playbook is a tool - use your judgment about what's relevant

Remember: The playbook provides general guidance. Adapt it to your specific situation.
"""

