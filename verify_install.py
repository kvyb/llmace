#!/usr/bin/env python3
"""
Quick verification script to check if llmace is properly installed.
"""

def main():
    print("=" * 60)
    print("LLMace Installation Verification")
    print("=" * 60)
    print()
    
    # Test 1: Import main package
    print("1. Testing main package import...")
    try:
        import llmace
        print(f"   ✅ llmace imported successfully (version {llmace.__version__})")
    except ImportError as e:
        print(f"   ❌ Failed to import llmace: {e}")
        return False
    
    # Test 2: Import main classes
    print("\n2. Testing main class imports...")
    try:
        from llmace import ACE, Bullet, ACEContext, ContextConfig, create_embedding_function
        print("   ✅ Main classes imported successfully")
    except ImportError as e:
        print(f"   ❌ Failed to import classes: {e}")
        return False
    
    # Test 3: Create ACE instance
    print("\n3. Testing ACE instantiation...")
    try:
        ace = ACE()
        print(f"   ✅ ACE instance created: {ace}")
    except Exception as e:
        print(f"   ❌ Failed to create ACE instance: {e}")
        return False
    
    # Test 4: Add a bullet
    print("\n4. Testing bullet addition...")
    try:
        bullet_id = ace.add_bullet(
            section="strategies",
            content="Test bullet for verification"
        )
        print(f"   ✅ Bullet added with ID: {bullet_id}")
    except Exception as e:
        print(f"   ❌ Failed to add bullet: {e}")
        return False
    
    # Test 5: Get playbook
    print("\n5. Testing playbook generation...")
    try:
        playbook = ace.get_playbook()
        if "Test bullet for verification" in playbook:
            print("   ✅ Playbook generated successfully")
        else:
            print("   ⚠️  Playbook generated but content missing")
    except Exception as e:
        print(f"   ❌ Failed to generate playbook: {e}")
        return False
    
    # Test 6: Serialization
    print("\n6. Testing serialization...")
    try:
        data = ace.to_dict()
        ace2 = ACE.from_dict(data)
        print(f"   ✅ Serialization working: {len(ace2.context)} bullets restored")
    except Exception as e:
        print(f"   ❌ Failed serialization: {e}")
        return False
    
    # Test 7: Check dependencies
    print("\n7. Checking dependencies...")
    try:
        import openai
        print("   ✅ openai installed")
    except ImportError:
        print("   ⚠️  openai not installed (optional for manual mode)")
    
    try:
        import pydantic
        print("   ✅ pydantic installed")
    except ImportError:
        print("   ❌ pydantic not installed (required)")
        return False
    
    try:
        import numpy
        print("   ✅ numpy installed")
    except ImportError:
        print("   ❌ numpy not installed (required)")
        return False
    
    # Optional: Check embeddings
    try:
        import sentence_transformers
        print("   ✅ sentence-transformers installed (embeddings support)")
    except ImportError:
        print("   ℹ️  sentence-transformers not installed (optional)")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! LLMace is properly installed.")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Try examples: python examples/basic_usage.py")
    print("  2. Read documentation: README.md")
    print("  3. Check quick start: QUICKSTART.md")
    print()
    
    return True


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)

