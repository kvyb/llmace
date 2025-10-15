"""
Auto-reflection example using OpenAI.

This example shows how to use LLMACE with automatic LLM-driven reflection and curation.
"""

import os
from openai import OpenAI
from llmace import LLMACE
from llmace.integrations import inject_playbook_into_messages


def main():
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-key'")
        return
    
    # Initialize OpenAI client for both LLM and embeddings
    client = OpenAI(api_key=api_key)
    
    # Initialize LLMACE with LLM client and embedding client for automatic reflection
    # When using OpenAI, you can use the same client for both
    llmace = LLMACE(
        llm_client=client,
        embedding_client=client,  # Same client for embeddings
        enable_logging=True
    )
    
    print("=" * 60)
    print("LLMACE Auto-Reflection Example")
    print("=" * 60)
    print()
    
    # Simulate a multi-turn workflow
    tasks = [
        {
            "query": "What is 15 * 24?",
            "expected": "360"
        },
        {
            "query": "Calculate the factorial of 5",
            "expected": "120"
        },
        {
            "query": "What is the square root of 144?",
            "expected": "12"
        }
    ]
    
    for i, task in enumerate(tasks, 1):
        print(f"\n{'=' * 60}")
        print(f"Task {i}/{len(tasks)}")
        print(f"{'=' * 60}")
        
        query = task["query"]
        print(f"Query: {query}")
        
        # Get current playbook
        playbook = llmace.get_playbook()
        
        # Create messages with playbook injected
        messages = [
            {"role": "system", "content": "You are a helpful math assistant."},
            {"role": "user", "content": query}
        ]
        
        if playbook:
            messages = inject_playbook_into_messages(
                messages=messages,
                context=llmace.context,
                position="system"
            )
        
        # Generate response
        response = client.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=messages,
            temperature=0
        )
        
        answer = response.choices[0].message.content
        print(f"Response: {answer}")
        
        # Check if correct
        expected = task["expected"]
        success = expected in answer
        print(f"Expected: {expected}")
        print(f"Success: {success}")
        
        # Reflect on execution (automatic mode - LLM will be called)
        print("\nReflecting on execution...")
        result = llmace.reflect(
            query=query,
            response=answer,
            success=success,
            feedback=f"Expected answer: {expected}",
            ground_truth=expected,
            auto_update=True,
            run_grow_and_refine=True
        )
        
        print(f"Reflection key insight: {result['reflection']['key_insight'][:100]}...")
        print(f"Generated {len(result['curation']['deltas'])} delta updates")
        print(f"Context stats: {result['update_stats']}")
    
    # Final playbook
    print(f"\n{'=' * 60}")
    print("Final Evolved Playbook:")
    print(f"{'=' * 60}")
    playbook = llmace.get_playbook(include_metadata=True)
    print(playbook)
    
    # Save final context
    filepath = "auto_reflection_context.json"
    llmace.save(filepath)
    print(f"\nContext saved to {filepath}")


if __name__ == "__main__":
    main()

