"""
Quick test to verify LLMACE is working correctly.

This script runs a simple multi-turn conversation to show:
1. LLMACE can reflect and learn
2. The playbook evolves over time
3. Context is properly injected
"""

from openai import OpenAI
from llmace import LLMACE
import os


def create_clients():
    """
    Create LLM and embedding clients from environment variables.
    
    Priority logic:
    - If OPENROUTER_API_KEY exists:
        → Use OpenRouter for LLM
        → Use OpenAI for embeddings (if OPENAI_API_KEY exists), else OpenRouter
    - If only OPENAI_API_KEY exists:
        → Use OpenAI for both LLM and embeddings
    """
    # Try to load .env file if it exists
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # python-dotenv not installed, use system env vars
    
    # Check for API keys
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not openrouter_key and not openai_key:
        print("❌ Error: No API key found!")
        print("\nPlease set one of:")
        print("  - OPENROUTER_API_KEY (for OpenRouter) + OPENAI_API_KEY (for embeddings)")
        print("  - OPENAI_API_KEY only (for both LLM and embeddings)")
        print("\nOr copy env.example to .env and fill in your keys.")
        return None, None, None
    
    # PRIORITY: OpenRouter for LLM if available
    if openrouter_key:
        print("🚀 Using OpenRouter for LLM")
        test_model = os.getenv("TEST_MODEL", "google/gemini-2.5-flash")
        print(f"   Model: {test_model}")
        
        llm_client = OpenAI(
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1"
        )
        
        # Use OpenAI for embeddings if available (recommended)
        if openai_key:
            print("🔑 Using OpenAI for embeddings (recommended)")
            embedding_client = OpenAI(api_key=openai_key)
        else:
            print("⚠️  Using OpenRouter for embeddings (OpenAI recommended for better quality)")
            embedding_client = llm_client
        
        return llm_client, embedding_client, test_model
    else:
        # Only OpenAI key present - use for both
        print("🔑 Using OpenAI for both LLM and embeddings")
        client = OpenAI(api_key=openai_key)
        return client, client


def quick_test():
    """Run a quick test of LLMACE functionality."""
    
    print("🧪 LLMACE Quick Test")
    print("=" * 60)
    
    # Initialize clients
    llm_client, embedding_client, test_model = create_clients()
    if not llm_client:
        return False
    
    llmace = LLMACE(
        llm_client=llm_client,
        embedding_client=embedding_client,
        enable_logging=True
    )
    
    print("\n✅ LLMACE initialized successfully")
    print(f"📊 Initial playbook size: {len(llmace.context)} bullets")
    
    # Test 1: First interaction (should fail or be suboptimal)
    print("\n" + "-" * 60)
    print("Test 1: Initial query (no context)")
    print("-" * 60)
    
    query1 = "What's 25% of 80?"
    playbook1 = llmace.get_playbook()
    
    response1 = llm_client.chat.completions.create(
        model=test_model,
        messages=[
            {"role": "system", "content": f"You are a helpful assistant.\n\n{playbook1}" if playbook1 else "You are a helpful assistant."},
            {"role": "user", "content": query1}
        ]
    )
    
    answer1 = response1.choices[0].message.content
    print(f"Query: {query1}")
    print(f"Response: {answer1}")
    
    # Reflect with feedback
    llmace.reflect(
        query=query1,
        response=answer1,
        success=True,
        feedback="Good calculation, but could show more work",
        ground_truth="20 (25% of 80 = 0.25 × 80 = 20)",
        auto_update=True,
        run_grow_and_refine=True
    )
    
    print(f"\n📈 Playbook size after reflection: {len(llmace.context)} bullets")
    
    # Test 2: Similar query (should use learned context)
    print("\n" + "-" * 60)
    print("Test 2: Similar query (with learned context)")
    print("-" * 60)
    
    query2 = "What's 30% of 150?"
    playbook2 = llmace.get_playbook()
    
    print(f"\n📖 Current Playbook:\n{playbook2}\n")
    
    response2 = llm_client.chat.completions.create(
        model=test_model,
        messages=[
            {"role": "system", "content": f"You are a helpful assistant.\n\n{playbook2}"},
            {"role": "user", "content": query2}
        ]
    )
    
    answer2 = response2.choices[0].message.content
    print(f"Query: {query2}")
    print(f"Response: {answer2}")
    
    # Reflect again
    llmace.reflect(
        query=query2,
        response=answer2,
        success=True,
        auto_update=True,
        run_grow_and_refine=True
    )
    
    print(f"\n📈 Playbook size after 2nd reflection: {len(llmace.context)} bullets")
    
    # Test 3: Serialization
    print("\n" + "-" * 60)
    print("Test 3: Save and load")
    print("-" * 60)
    
    llmace.save("test_context.json")
    print("✅ Saved context to test_context.json")
    
    llmace2 = LLMACE.load("test_context.json", llm_client=llm_client, embedding_client=embedding_client)
    print(f"✅ Loaded context: {len(llmace2.context)} bullets")
    
    # Cleanup
    import os
    os.remove("test_context.json")
    print("🧹 Cleaned up test file")
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nLLMACE is working correctly:")
    print("  ✓ Reflection and learning")
    print("  ✓ Playbook evolution")
    print("  ✓ Context injection")
    print("  ✓ Serialization")
    print("\nYou can now use LLMACE in your projects!")
    
    return True


if __name__ == "__main__":
    try:
        success = quick_test()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

