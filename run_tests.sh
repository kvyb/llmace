#!/bin/bash

# LLMACE Test Runner
# Run this to test LLMACE functionality

set -e

echo "========================================================================"
echo "LLMACE TEST SUITE"
echo "========================================================================"
echo ""

# Get script directory (root of llmace)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate venv if exists
if [ -d "$SCRIPT_DIR/.venv" ]; then
    echo "✓ Activating .venv"
    source "$SCRIPT_DIR/.venv/bin/activate"
else
    echo "⚠️  No .venv found (optional)"
fi

# Load .env file from root
if [ -f "$SCRIPT_DIR/.env" ]; then
    echo "✓ Loading .env from root"
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
else
    echo "⚠️  No .env file found in root"
fi

echo ""

# Check for API keys
if [ -z "$OPENROUTER_API_KEY" ] && [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Error: No API keys found!"
    echo ""
    echo "Make sure your .env file contains:"
    echo "  OPENROUTER_API_KEY=sk-or-v1-..."
    echo "  # or"
    echo "  OPENAI_API_KEY=sk-..."
    echo ""
    exit 1
fi

echo "✓ API keys configured"
echo ""

# Ask which test to run
echo "Select test to run:"
echo "  1) Quick Test (sanity check)"
echo "  2) Feature Coverage (all features)"
echo "  3) LangGraph Integration (production demo)"
echo "  4) Benchmark Suite (comprehensive comparison)"
echo "  5) FAQ Benchmark (business context learning - 50 questions)"
echo "  6) All tests (sequential)"
echo ""
read -p "Enter choice [1-6]: " choice

case $choice in
    1)
        echo ""
        echo "Running Quick Test..."
        python tests/quick_test.py
        ;;
    2)
        echo ""
        echo "Running Feature Coverage Test..."
        python tests/test_feature_coverage.py
        ;;
    3)
        echo ""
        echo "Running LangGraph Integration Test..."
        python tests/test_langgraph_integration.py
        ;;
    4)
        echo ""
        echo "Running Benchmark Suite (this will take several minutes)..."
        python tests/benchmark_suite.py
        ;;
    5)
        echo ""
        echo "Running FAQ Benchmark (this will take 10-15 minutes)..."
        python tests/benchmark_faq.py
        ;;
    6)
        echo ""
        echo "Running All Tests..."
        echo ""
        echo "1/5: Quick Test"
        python tests/quick_test.py
        echo ""
        echo "2/5: Feature Coverage"
        python tests/test_feature_coverage.py
        echo ""
        echo "3/5: LangGraph Integration"
        python tests/test_langgraph_integration.py
        echo ""
        echo "4/5: Benchmark Suite"
        python tests/benchmark_suite.py
        echo ""
        echo "5/5: FAQ Benchmark"
        python tests/benchmark_faq.py
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "========================================================================"
echo "✓ Tests Complete!"
echo "========================================================================"

