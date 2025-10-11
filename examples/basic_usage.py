"""
Basic ACE usage example - Manual reflection mode.

This example shows how to use ACE without automatic LLM calls,
by manually providing reflection and curation results.
"""

from llmace import ACE

def main():
    # Initialize ACE without LLM client (manual mode)
    ace = ACE(enable_logging=True)
    
    print("=" * 60)
    print("ACE Basic Usage Example - Manual Mode")
    print("=" * 60)
    print()
    
    # Simulate a task execution
    query = "Calculate the sum of numbers from 1 to 100"
    response = "The sum is 5050 using the formula n*(n+1)/2"
    success = True
    
    print(f"Query: {query}")
    print(f"Response: {response}")
    print(f"Success: {success}")
    print()
    
    # Manually provide reflection result
    reflection_result = {
        "reasoning": "The agent correctly applied the arithmetic series formula.",
        "error_identification": None,
        "root_cause_analysis": None,
        "correct_approach": None,
        "key_insight": "For arithmetic series, use the formula n*(n+1)/2 instead of iterating",
        "bullet_tags": []
    }
    
    # Manually provide curation result
    curation_result = {
        "reasoning": "Adding the arithmetic series formula as a reusable strategy",
        "deltas": [
            {
                "operation": "add",
                "bullet_id": None,
                "section": "strategies",
                "content": "For arithmetic series sum, use formula n*(n+1)/2 for efficiency",
                "metadata": {}
            },
            {
                "operation": "add",
                "bullet_id": None,
                "section": "best_practices",
                "content": "Prefer mathematical formulas over iteration when available",
                "metadata": {}
            }
        ]
    }
    
    # Reflect and update context
    print("Running reflection and curation...")
    result = ace.reflect(
        query=query,
        response=response,
        success=success,
        reflection_result=reflection_result,
        curation_result=curation_result,
        auto_update=True
    )
    
    print(f"Context updated: {result['update_stats']}")
    print()
    
    # Get the evolved playbook
    print("=" * 60)
    print("Current Playbook:")
    print("=" * 60)
    playbook = ace.get_playbook(include_metadata=True)
    print(playbook)
    print()
    
    # Save context
    filepath = "basic_context.json"
    ace.save(filepath)
    print(f"Context saved to {filepath}")
    print()
    
    # Load context
    ace_loaded = ACE.load(filepath)
    print(f"Context loaded: {ace_loaded}")
    print()
    
    # Show statistics
    print("=" * 60)
    print("Context Statistics:")
    print("=" * 60)
    print(f"Total bullets: {len(ace.context)}")
    for section in ace.context.config.sections:
        bullets = ace.context.get_bullets_by_section(section)
        print(f"  {section}: {len(bullets)} bullets")

if __name__ == "__main__":
    main()

