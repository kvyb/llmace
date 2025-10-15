"""
OpenRouter integration example.

This example shows how to use LLMACE with OpenRouter to access various LLM providers.
"""

import os
from openai import OpenAI
from llmace import LLMACE


def main():
    # Check for API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        print("Get your key from: https://openrouter.ai/keys")
        print("Set it with: export OPENROUTER_API_KEY='your-key'")
        return
    
    # Initialize OpenAI client with OpenRouter base URL for LLM
    llm_client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
    
    # Set default model (you can change this to any OpenRouter model)
    llm_client.default_model = "anthropic/claude-3.5-sonnet"
    
    print("=" * 60)
    print("LLMACE with OpenRouter Example")
    print(f"Using LLM model: {llm_client.default_model}")
    print("=" * 60)
    print()
    
    # For embeddings, we need a separate OpenAI client
    # (OpenRouter doesn't support embedding models)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    embedding_client = None
    if openai_api_key:
        print("OpenAI API key found - enabling embeddings for deduplication")
        embedding_client = OpenAI(api_key=openai_api_key)
    else:
        print("No OpenAI API key found - deduplication will be disabled")
        print("To enable: export OPENAI_API_KEY='your-key'")
    
    print()
    
    # Initialize LLMACE with separate clients
    llmace = LLMACE(
        llm_client=llm_client,
        embedding_client=embedding_client,  # Separate OpenAI client for embeddings
        enable_logging=True
    )
    
    # Example task
    query = "Explain the concept of recursion in programming"
    print(f"Query: {query}\n")
    
    # Generate response using OpenRouter
    messages = [
        {"role": "system", "content": "You are a helpful programming tutor."},
        {"role": "user", "content": query}
    ]
    
    print("Generating response...")
    response = llm_client.chat.completions.create(
        model=llm_client.default_model,
        messages=messages
    )
    
    answer = response.choices[0].message.content
    print(f"\nResponse: {answer[:200]}...\n")
    
    # Reflect on the execution
    print("Reflecting on execution...")
    result = llmace.reflect(
        query=query,
        response=answer,
        success=True,
        feedback="Good explanation with examples",
        auto_update=True,
        run_grow_and_refine=True
    )
    
    print(f"Key insight: {result['reflection']['key_insight']}")
    print(f"Context updated: {result['update_stats']}")
    print()
    
    # Show evolved playbook
    print("=" * 60)
    print("Evolved Playbook:")
    print("=" * 60)
    playbook = llmace.get_playbook()
    print(playbook if playbook else "[Empty playbook]")
    
    # Save context
    filepath = "openrouter_context.json"
    llmace.save(filepath)
    print(f"\nContext saved to {filepath}")
    
    # Demonstrate loading and reusing context
    print("\n" + "=" * 60)
    print("Loading context for reuse...")
    print("=" * 60)
    
    llmace_loaded = LLMACE.load(
        filepath,
        llm_client=llm_client,
        embedding_client=embedding_client
    )
    print(f"Loaded context: {llmace_loaded}")
    print(f"Total bullets: {len(llmace_loaded.context)}")


if __name__ == "__main__":
    main()

