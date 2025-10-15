"""
Test LLMACE effectiveness using LLM-as-a-judge evaluation.

This test compares performance with and without LLMACE across multiple iterations
to demonstrate that the evolving context actually improves LLM performance.
"""

import json
from typing import Dict, List
from openai import OpenAI
from llmace import LLMACE


class LLMJudge:
    """LLM-as-a-judge for evaluating response quality."""
    
    def __init__(self, client: OpenAI, model: str = "google/gemini-2.5-flash"):
        self.client = client
        self.model = model
    
    def evaluate(self, query: str, response: str, ground_truth: str = None) -> Dict:
        """
        Evaluate a response using LLM as judge.
        
        Returns:
            Dict with score (0-10), reasoning, and correctness
        """
        prompt = f"""You are an expert evaluator. Evaluate the following response to a query.

**Query:**
{query}

**Response:**
{response}

{f"**Ground Truth/Expected Answer:**\n{ground_truth}\n" if ground_truth else ""}

**Evaluation Criteria:**
1. Correctness: Is the answer correct?
2. Completeness: Does it fully address the query?
3. Clarity: Is it well-explained and easy to understand?
4. Relevance: Does it stay on topic?

**Output Format:**
Respond with a JSON object:
{{
    "score": <0-10>,
    "correctness": <true/false>,
    "reasoning": "Brief explanation of the score",
    "strengths": "What was good",
    "weaknesses": "What could be improved"
}}
"""
        
        result = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert evaluator. Respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        return json.loads(result.choices[0].message.content)


class LLMACEEffectivenessTest:
    """Test framework for evaluating LLMACE effectiveness."""
    
    def __init__(self, client: OpenAI, embedding_client: OpenAI = None, test_model: str = "google/gemini-2.5-flash", judge_model: str = "google/gemini-2.5-flash"):
        self.client = client
        self.embedding_client = embedding_client or client
        self.test_model = test_model
        self.judge = LLMJudge(client, model=judge_model)
    
    def run_baseline(self, tasks: List[Dict]) -> List[Dict]:
        """
        Run tasks without LLMACE (baseline).
        
        Args:
            tasks: List of dicts with 'query' and optional 'ground_truth'
        
        Returns:
            List of results with responses and evaluations
        """
        results = []
        
        for i, task in enumerate(tasks):
            print(f"\n[Baseline] Task {i+1}/{len(tasks)}: {task['query'][:50]}...")
            
            # Generate response without LLMACE
            response = self.client.chat.completions.create(
                model=self.test_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": task["query"]}
                ],
                temperature=0.7
            )
            
            answer = response.choices[0].message.content
            
            # Evaluate
            evaluation = self.judge.evaluate(
                task["query"],
                answer,
                task.get("ground_truth")
            )
            
            results.append({
                "task": task,
                "response": answer,
                "evaluation": evaluation
            })
            
            print(f"  Score: {evaluation['score']}/10")
        
        return results
    
    def run_with_llmace(self, tasks: List[Dict]) -> List[Dict]:
        """
        Run tasks with LLMACE (with learning).
        
        Args:
            tasks: List of dicts with 'query' and optional 'ground_truth'
        
        Returns:
            List of results with responses and evaluations
        """
        # Initialize LLMACE
        llmace = LLMACE(
            llm_client=self.client,
            embedding_client=self.embedding_client,
            enable_logging=False
        )
        
        results = []
        
        for i, task in enumerate(tasks):
            print(f"\n[LLMACE] Task {i+1}/{len(tasks)}: {task['query'][:50]}...")
            
            # Get current playbook
            playbook = llmace.get_playbook()
            
            # Generate response with LLMACE playbook
            system_prompt = "You are a helpful assistant."
            if playbook:
                system_prompt += f"\n\n**Accumulated Knowledge:**\n{playbook}"
            
            response = self.client.chat.completions.create(
                model=self.test_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": task["query"]}
                ],
                temperature=0.7
            )
            
            answer = response.choices[0].message.content
            
            # Evaluate
            evaluation = self.judge.evaluate(
                task["query"],
                answer,
                task.get("ground_truth")
            )
            
            # Let LLMACE learn from this execution
            llmace.reflect(
                query=task["query"],
                response=answer,
                success=evaluation["correctness"],
                feedback=evaluation["reasoning"],
                ground_truth=task.get("ground_truth"),
                auto_update=True,
                run_grow_and_refine=True
            )
            
            results.append({
                "task": task,
                "response": answer,
                "evaluation": evaluation,
                "playbook_size": len(llmace.context)
            })
            
            print(f"  Score: {evaluation['score']}/10 | Playbook: {len(llmace.context)} bullets")
        
        return results
    
    def compare_results(self, baseline_results: List[Dict], llmace_results: List[Dict]) -> Dict:
        """
        Compare baseline vs LLMACE results.
        
        Returns:
            Dict with comparison metrics
        """
        baseline_scores = [r["evaluation"]["score"] for r in baseline_results]
        llmace_scores = [r["evaluation"]["score"] for r in llmace_results]
        
        baseline_correct = sum(1 for r in baseline_results if r["evaluation"]["correctness"])
        llmace_correct = sum(1 for r in llmace_results if r["evaluation"]["correctness"])
        
        return {
            "baseline": {
                "avg_score": sum(baseline_scores) / len(baseline_scores),
                "min_score": min(baseline_scores),
                "max_score": max(baseline_scores),
                "correct_count": baseline_correct,
                "accuracy": baseline_correct / len(baseline_results)
            },
            "llmace": {
                "avg_score": sum(llmace_scores) / len(llmace_scores),
                "min_score": min(llmace_scores),
                "max_score": max(llmace_scores),
                "correct_count": llmace_correct,
                "accuracy": llmace_correct / len(llmace_results)
            },
            "improvement": {
                "avg_score_delta": (sum(llmace_scores) - sum(baseline_scores)) / len(baseline_scores),
                "accuracy_delta": (llmace_correct - baseline_correct) / len(baseline_results),
                "percentage_improvement": ((sum(llmace_scores) - sum(baseline_scores)) / sum(baseline_scores)) * 100
            }
        }
    
    def print_report(self, comparison: Dict):
        """Print a formatted comparison report."""
        print("\n" + "=" * 70)
        print("LLMACE EFFECTIVENESS TEST RESULTS")
        print("=" * 70)
        
        print("\n📊 BASELINE (No LLMACE):")
        print(f"  Average Score: {comparison['baseline']['avg_score']:.2f}/10")
        print(f"  Accuracy: {comparison['baseline']['accuracy']*100:.1f}% ({comparison['baseline']['correct_count']} correct)")
        print(f"  Score Range: {comparison['baseline']['min_score']}-{comparison['baseline']['max_score']}")
        
        print("\n🚀 WITH LLMACE:")
        print(f"  Average Score: {comparison['llmace']['avg_score']:.2f}/10")
        print(f"  Accuracy: {comparison['llmace']['accuracy']*100:.1f}% ({comparison['llmace']['correct_count']} correct)")
        print(f"  Score Range: {comparison['llmace']['min_score']}-{comparison['llmace']['max_score']}")
        
        print("\n📈 IMPROVEMENT:")
        print(f"  Score Delta: +{comparison['improvement']['avg_score_delta']:.2f} points")
        print(f"  Accuracy Delta: +{comparison['improvement']['accuracy_delta']*100:.1f}%")
        print(f"  Percentage Improvement: {comparison['improvement']['percentage_improvement']:.1f}%")
        
        if comparison['improvement']['avg_score_delta'] > 0:
            print("\n✅ LLMACE shows positive improvement!")
        else:
            print("\n⚠️  No significant improvement detected")
        
        print("=" * 70)


def create_test_tasks() -> List[Dict]:
    """
    Create a set of test tasks for evaluation.
    
    These tasks should be challenging enough that learning helps,
    but solvable by the LLM.
    """
    return [
        {
            "query": "Calculate the compound interest on $1000 at 5% annual rate for 3 years, compounded annually.",
            "ground_truth": "The formula is A = P(1 + r)^t. So A = 1000(1.05)^3 = $1157.63. The interest is $157.63."
        },
        {
            "query": "Convert 72 kilometers per hour to meters per second.",
            "ground_truth": "72 km/h = 72000 m/h = 72000/3600 m/s = 20 m/s"
        },
        {
            "query": "What is the area of a circle with diameter 14 cm?",
            "ground_truth": "Radius = 7 cm. Area = π × r² = π × 49 ≈ 153.94 cm²"
        },
        {
            "query": "If a train travels 240 km in 3 hours, what is its average speed?",
            "ground_truth": "Average speed = distance/time = 240/3 = 80 km/h"
        },
        {
            "query": "How many minutes are there in a week?",
            "ground_truth": "7 days × 24 hours × 60 minutes = 10,080 minutes"
        },
        {
            "query": "Calculate 15% of 80.",
            "ground_truth": "15% of 80 = 0.15 × 80 = 12"
        },
        {
            "query": "What is the perimeter of a rectangle with length 12 cm and width 8 cm?",
            "ground_truth": "Perimeter = 2(l + w) = 2(12 + 8) = 40 cm"
        },
        {
            "query": "Convert 5 pounds to kilograms (1 lb ≈ 0.45 kg).",
            "ground_truth": "5 × 0.45 = 2.25 kg"
        },
    ]


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
    import os
    
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
        return None, None
    
    # PRIORITY: OpenRouter for LLM if available
    if openrouter_key:
        print("🚀 Using OpenRouter for LLM")
        test_model = os.getenv("TEST_MODEL", "x-ai/grok-2-fast")
        judge_model = os.getenv("JUDGE_MODEL", "google/gemini-2.5-flash")
        
        print(f"   Test Model: {test_model}")
        print(f"   Judge Model: {judge_model}")
        
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
            embedding_client = OpenAI(
                api_key=openrouter_key,
                base_url="https://openrouter.ai/api/v1"
            )
    else:
        # Only OpenAI key present - use for both
        print("🔑 Using OpenAI for both LLM and embeddings")
        llm_client = OpenAI(api_key=openai_key)
        embedding_client = OpenAI(api_key=openai_key)
    
    return llm_client, embedding_client


def main():
    """Run the LLMACE effectiveness test."""
    import os
    
    print("=" * 70)
    print("LLMACE EFFECTIVENESS TEST")
    print("=" * 70)
    print("\nThis test compares LLM performance with and without LLMACE.")
    print("Using LLM-as-a-judge for evaluation.\n")
    
    # Initialize clients
    llm_client, embedding_client = create_clients()
    if not llm_client:
        return
    
    # Get model names from environment
    # Support multiple test models (comma-separated)
    test_models_str = os.getenv("TEST_MODELS", "")
    if test_models_str and "," in test_models_str:
        test_models = [m.strip() for m in test_models_str.split(",")]
        print(f"\n🔄 Multi-model testing enabled: {len(test_models)} models")
        print(f"   Models: {', '.join(test_models)}\n")
    else:
        test_models = [os.getenv("TEST_MODEL", "x-ai/grok-2-fast" if os.getenv("OPENROUTER_API_KEY") else "google/gemini-2.5-flash")]
    
    judge_model = os.getenv("JUDGE_MODEL", "google/gemini-2.5-flash")
    print(f"🎯 Judge Model: {judge_model}\n")
    
    # Create test tasks
    tasks = create_test_tasks()
    print(f"Created {len(tasks)} test tasks\n")
    
    all_results = {}
    
    # Test each model
    for model in test_models:
        print("\n" + "=" * 70)
        print(f"TESTING WITH MODEL: {model}")
        print("=" * 70)
        
        tester = LLMACEEffectivenessTest(llm_client, embedding_client, test_model=model, judge_model=judge_model)
        
        # Run baseline
        print("\n" + "-" * 70)
        print("PHASE 1: Running baseline (no LLMACE)...")
        print("-" * 70)
        baseline_results = tester.run_baseline(tasks)
        
        # Run with LLMACE
        print("\n" + "-" * 70)
        print("PHASE 2: Running with LLMACE (with learning)...")
        print("-" * 70)
        llmace_results = tester.run_with_llmace(tasks)
        
        # Compare and report
        comparison = tester.compare_results(baseline_results, llmace_results)
        tester.print_report(comparison)
        
        # Store results
        all_results[model] = {
            "comparison": comparison,
            "baseline_results": baseline_results,
            "llmace_results": llmace_results
        }
    
    # If multiple models, print comparison summary
    if len(test_models) > 1:
        print("\n" + "=" * 70)
        print("MULTI-MODEL COMPARISON SUMMARY")
        print("=" * 70)
        for model in test_models:
            comp = all_results[model]["comparison"]
            print(f"\n📊 {model}:")
            print(f"   Improvement: {comp['improvement']['percentage_improvement']:.1f}%")
            print(f"   Score Delta: +{comp['improvement']['avg_score_delta']:.2f}")
            print(f"   LLMACE Avg Score: {comp['llmace']['avg_score']:.2f}/10")
    
    # Save detailed results
    results_file = "llmace_test_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n📁 Detailed results saved to: {results_file}")


if __name__ == "__main__":
    main()

