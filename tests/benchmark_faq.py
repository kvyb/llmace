#!/usr/bin/env python3
"""
FAQ Agent Benchmark - Testing LLMACE Progressive Learning on Business Context

This benchmark demonstrates LLMACE's ability to learn business-specific patterns
over time by organizing questions into semantic clusters and tracking performance
improvement as the agent encounters similar questions.

Key Features:
- Semantic clustering of questions (pricing, security, technical, etc.)
- Progressive learning tracking (early vs mid vs late performance)
- Faithfulness to business context evaluation
- Appropriate handoff decision tracking
"""

import os
import json
import time
import statistics
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from openai import OpenAI
from llmace import LLMACE
import backoff

try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass


# ============================================================================
# Business Context
# ============================================================================

BUSINESS_CONTEXT = """
# CloudSync SaaS Platform - Business Context

## Company Overview
CloudSync is a B2B file synchronization and collaboration platform. 
Founded in 2020, serving 5,000+ businesses.

## Pricing Plans
- **Starter**: $9/user/month - 100GB storage, 30-day retention, email support
- **Professional**: $29/user/month - 1TB storage, 1-year retention, REST API, priority support
- **Enterprise**: Custom pricing - Unlimited storage, unlimited retention, REST + GraphQL API, 24/7 support

## Key Features by Plan
- File size limit: 5GB per file (all plans)
- Team size: Unlimited users (all plans)
- Real-time collaboration: Professional+ only (launched Dec 2023)
- Migration tool: Available from Dropbox/Google Drive (all plans)

## Billing & Policies
- Billing: Monthly or annual (10% discount on annual)
- Charges: 1st of each month
- Cancellation: Anytime with prorated refunds on annual plans only
- SLA: 99.5% (Professional), 99.9% (Enterprise)

## Security & Compliance
- Encryption: AES-256
- Certifications: SOC 2 Type II, GDPR compliant
- API rate limit: 5000 requests/hour (increased from 1000 in Jan 2024)

## Known Issues
- Mobile push notifications delayed (fix ETA: 2 weeks)
- Safari upload issues for files >2GB (workaround: use Chrome)
- Linux sync client in beta (not production-ready)

## Contact Points
- Sales: sales@cloudsync.io
- Support: support@cloudsync.io
- Billing: billing@cloudsync.io
- Status: status.cloudsync.io
"""


# ============================================================================
# Semantic Question Clusters
# ============================================================================

QUESTION_CLUSTERS = {
    "pricing_basic": [
        {"id": 1, "q": "How much does CloudSync cost?", "answerable": True, "expect": ["$9", "$29", "Enterprise custom"]},
        {"id": 2, "q": "What's included in the Starter plan?", "answerable": True, "expect": ["$9", "100GB", "30-day", "email support"]},
        {"id": 3, "q": "Do you offer annual discounts?", "answerable": True, "expect": ["10% discount", "annual"]},
        {"id": 4, "q": "What's the difference between Professional and Enterprise?", "answerable": True, "expect": ["storage", "API", "support", "SLA"]},
        {"id": 5, "q": "Can students get a discount?", "answerable": False, "handoff": True, "expect": ["sales@cloudsync.io"]},
    ],
    
    "pricing_advanced": [
        {"id": 6, "q": "I need API access but I'm on Starter. What should I do?", "answerable": True, "expect": ["upgrade", "Professional", "$29"]},
        {"id": 7, "q": "If I upgrade mid-month, do I pay full price?", "answerable": False, "handoff": True, "expect": ["billing@cloudsync.io"]},
        {"id": 8, "q": "What happens to my data if I cancel?", "answerable": True, "expect": ["retention", "30-day or 1-year depending on plan"]},
        {"id": 9, "q": "Can I get a custom Enterprise plan for a nonprofit?", "answerable": False, "handoff": True, "expect": ["sales@cloudsync.io"]},
        {"id": 10, "q": "How many users can I have on each plan?", "answerable": True, "expect": ["unlimited", "all plans"]},
    ],
    
    "security_basic": [
        {"id": 11, "q": "How do you encrypt my data?", "answerable": True, "expect": ["AES-256", "encrypted"]},
        {"id": 12, "q": "Are you GDPR compliant?", "answerable": True, "expect": ["GDPR compliant", "SOC 2"]},
        {"id": 13, "q": "Where are your data centers located?", "answerable": False, "handoff": True, "expect": ["not in context"]},
        {"id": 14, "q": "Do you support two-factor authentication?", "answerable": False, "handoff": False, "expect": ["not specified"]},
        {"id": 15, "q": "What certifications do you have?", "answerable": True, "expect": ["SOC 2 Type II", "GDPR"]},
    ],
    
    "security_advanced": [
        {"id": 16, "q": "Can I audit who accessed my files?", "answerable": False, "handoff": True, "expect": ["not in context"]},
        {"id": 17, "q": "Do you encrypt data at rest and in transit?", "answerable": True, "expect": ["AES-256", "encryption"]},
        {"id": 18, "q": "What's your security incident response process?", "answerable": False, "handoff": True, "expect": ["support@cloudsync.io"]},
        {"id": 19, "q": "Are you SOC 2 certified?", "answerable": True, "expect": ["SOC 2 Type II", "certified"]},
        {"id": 20, "q": "Can I set up SSO for my team?", "answerable": False, "handoff": True, "expect": ["Enterprise feature"]},
    ],
    
    "technical_basic": [
        {"id": 21, "q": "What's the maximum file size I can upload?", "answerable": True, "expect": ["5GB", "per file"]},
        {"id": 22, "q": "Do you have an API?", "answerable": True, "expect": ["Professional: REST", "Enterprise: GraphQL"]},
        {"id": 23, "q": "What's your API rate limit?", "answerable": True, "expect": ["5000 requests per hour"]},
        {"id": 24, "q": "Do you support Linux?", "answerable": True, "expect": ["Linux", "beta", "not production"]},
        {"id": 25, "q": "What's your uptime guarantee?", "answerable": True, "expect": ["99.5%", "99.9%", "SLA"]},
    ],
    
    "technical_advanced": [
        {"id": 26, "q": "Can I use webhooks with your API?", "answerable": False, "handoff": True, "expect": ["not in context"]},
        {"id": 27, "q": "Do you have a Python SDK?", "answerable": False, "handoff": True, "expect": ["not specified"]},
        {"id": 28, "q": "Large files fail in Safari. Is this a known issue?", "answerable": True, "expect": ["Safari", ">2GB", "Chrome workaround"]},
        {"id": 29, "q": "Can I get real-time collaboration on Starter plan?", "answerable": True, "expect": ["Professional+ only", "upgrade needed"]},
        {"id": 30, "q": "What programming languages does your API support?", "answerable": False, "handoff": False, "expect": ["REST API"]},
    ],
    
    "billing_basic": [
        {"id": 31, "q": "When do you charge my credit card?", "answerable": True, "expect": ["1st of month"]},
        {"id": 32, "q": "Can I cancel anytime?", "answerable": True, "expect": ["cancel anytime", "prorated refund"]},
        {"id": 33, "q": "What payment methods do you accept?", "answerable": False, "handoff": True, "expect": ["billing@cloudsync.io"]},
        {"id": 34, "q": "Do you offer refunds?", "answerable": True, "expect": ["prorated", "annual only"]},
        {"id": 35, "q": "Can I pay annually?", "answerable": True, "expect": ["annual", "10% discount"]},
    ],
    
    "billing_issues": [
        {"id": 36, "q": "I was charged twice this month. Why?", "answerable": False, "handoff": True, "expect": ["billing@cloudsync.io", "urgent"]},
        {"id": 37, "q": "How do I update my credit card?", "answerable": False, "handoff": True, "expect": ["billing@cloudsync.io"]},
        {"id": 38, "q": "Can I get an invoice for my payment?", "answerable": False, "handoff": True, "expect": ["billing@cloudsync.io"]},
        {"id": 39, "q": "What happens if my payment fails?", "answerable": False, "handoff": True, "expect": ["billing@cloudsync.io"]},
        {"id": 40, "q": "Can I downgrade my plan?", "answerable": False, "handoff": True, "expect": ["billing@cloudsync.io"]},
    ],
    
    "support_migration": [
        {"id": 41, "q": "I'm migrating from Dropbox. Can you help?", "answerable": True, "expect": ["migration tool", "Dropbox", "all plans"]},
        {"id": 42, "q": "How long does migration take?", "answerable": False, "handoff": False, "expect": ["depends on data"]},
        {"id": 43, "q": "Can I migrate from Google Drive?", "answerable": True, "expect": ["migration tool", "Google Drive"]},
        {"id": 44, "q": "Will my folder structure be preserved?", "answerable": False, "handoff": False, "expect": ["not specified"]},
        {"id": 45, "q": "My files are missing after sync!", "answerable": False, "handoff": True, "expect": ["support@cloudsync.io", "urgent"]},
    ],
    
    "support_issues": [
        {"id": 46, "q": "Push notifications aren't working on my phone. Known issue?", "answerable": True, "expect": ["known issue", "delayed", "fix ETA 2 weeks"]},
        {"id": 47, "q": "Where can I check if CloudSync is down?", "answerable": True, "expect": ["status.cloudsync.io"]},
        {"id": 48, "q": "How do I contact support?", "answerable": True, "expect": ["support@cloudsync.io", "plan-specific"]},
        {"id": 49, "q": "Can I get a dedicated account manager?", "answerable": True, "expect": ["Enterprise only"]},
        {"id": 50, "q": "My sync is stuck. What should I do?", "answerable": False, "handoff": True, "expect": ["support@cloudsync.io"]},
    ],
}


# ============================================================================
# Metrics
# ============================================================================

@dataclass
class ClusterMetrics:
    """Metrics for a question cluster."""
    cluster_name: str
    questions_count: int
    avg_correctness: float
    avg_completeness: float
    avg_faithfulness: float
    handoff_accuracy: float  # % of correct handoff decisions
    
    def to_dict(self):
        return asdict(self)


@dataclass
class ProgressionMetrics:
    """Tracks learning progression over time."""
    early_correctness: float  # First 20 questions
    mid_correctness: float    # Questions 21-35
    late_correctness: float   # Questions 36-50
    
    improvement: float  # late - early
    
    def to_dict(self):
        return asdict(self)


# ============================================================================
# FAQ Agent
# ============================================================================

class FAQAgent:
    """FAQ agent with optional LLMACE."""
    
    def __init__(self, llm_client: OpenAI, model: str, use_llmace: bool = False, llmace: LLMACE = None):
        self.llm_client = llm_client
        self.model = model
        self.use_llmace = use_llmace
        self.llmace = llmace
    
    def answer_question(self, question: str) -> str:
        """Answer a FAQ question."""
        system_prompt = self._build_system_prompt()
        
        @backoff.on_exception(backoff.expo, Exception, max_tries=3, max_time=30)
        def call_llm():
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=0.3,
                max_tokens=400
            )
            return response.choices[0].message.content
        
        return call_llm()
    
    def _build_system_prompt(self) -> str:
        """Build system prompt."""
        prompt = f"""You are a customer support agent for CloudSync.

{BUSINESS_CONTEXT}

**Instructions:**
- Answer accurately using the business context
- Be concise and helpful
- If information is missing, say "I'll escalate this to [team]@cloudsync.io"
- Never make up information

"""
        if self.use_llmace and self.llmace:
            playbook = self.llmace.get_playbook()
            if playbook:
                prompt += f"\n**Learned Best Practices:**\n{playbook}\n"
        
        return prompt


# ============================================================================
# Judge
# ============================================================================

class FAQJudge:
    """Evaluates FAQ responses."""
    
    def __init__(self, llm_client: OpenAI, judge_model: str):
        self.llm_client = llm_client
        self.judge_model = judge_model
    
    def evaluate(self, q_data: Dict, answer: str) -> Dict[str, Any]:
        """Evaluate response."""
        
        json_schema = {
            "name": "evaluation",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "correctness": {"type": "number"},
                    "completeness": {"type": "number"},
                    "faithfulness": {"type": "number"},
                    "handoff_correct": {"type": "boolean"},
                    "reasoning": {"type": "string"}
                },
                "required": ["correctness", "completeness", "faithfulness", "handoff_correct", "reasoning"],
                "additionalProperties": False
            }
        }
        
        # Check if agent handed off
        handoff_detected = any(phrase in answer.lower() for phrase in [
            "escalate", "contact support", "contact sales", "contact billing",
            "@cloudsync.io", "i'll need to", "i don't have"
        ])
        
        should_handoff = q_data.get("handoff", False)
        
        prompt = f"""Evaluate this FAQ response.

**Business Context:**
{BUSINESS_CONTEXT}

**Question:** {q_data['q']}
**Answerable:** {q_data['answerable']}
**Expected Info:** {q_data.get('expect', 'N/A')}
**Should Handoff:** {should_handoff}

**Agent Answer:** {answer}
**Agent Handed Off:** {handoff_detected}

**Evaluate (0-1 scale):**
1. **Correctness**: Accurate per business context?
2. **Completeness**: Covered key points?
3. **Faithfulness**: No hallucinations?
4. **Handoff Correct**: Appropriate handoff decision?

Respond with JSON only."""
        
        @backoff.on_exception(backoff.expo, Exception, max_tries=3, max_time=30)
        def call_judge():
            response = self.llm_client.chat.completions.create(
                model=self.judge_model,
                messages=[
                    {"role": "system", "content": "Expert evaluator. JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_schema", "json_schema": json_schema}
            )
            return json.loads(response.choices[0].message.content)
        
        return call_judge()


# ============================================================================
# Benchmark
# ============================================================================

class FAQBenchmark:
    """Run FAQ benchmark with learning progression tracking."""
    
    def __init__(self, llm_client: OpenAI, embedding_client: OpenAI, model: str, judge_model: str):
        self.llm_client = llm_client
        self.embedding_client = embedding_client
        self.model = model
        self.judge = FAQJudge(llm_client, judge_model)
    
    def run(self, use_llmace: bool = False) -> Tuple[Dict[str, ClusterMetrics], ProgressionMetrics, List[Dict]]:
        """Run benchmark."""
        
        method_name = "LLMACE" if use_llmace else "Baseline"
        print(f"\n{'='*80}")
        print(f"{'🚀 ' + method_name if use_llmace else '📊 ' + method_name}")
        print(f"{'='*80}\n")
        
        # Setup
        llmace = None
        if use_llmace:
            from llmace.core.schemas import ContextConfig
            config = ContextConfig(
                max_bullets_per_section=20,  # Force refinement, not just growth
                dedup_threshold=0.85
            )
            llmace = LLMACE(
                llm_client=self.llm_client,
                embedding_client=self.embedding_client,
                config=config,
                enable_logging=False  # Less verbose for 50 questions
            )
        
        agent = FAQAgent(self.llm_client, self.model, use_llmace, llmace)
        
        # Track results
        all_results = []
        cluster_results = {}
        
        # Flatten questions in cluster order
        question_id = 0
        for cluster_name, questions in QUESTION_CLUSTERS.items():
            print(f"\n📂 Cluster: {cluster_name}")
            print("-" * 60)
            
            cluster_scores = {
                "correctness": [],
                "completeness": [],
                "faithfulness": [],
                "handoff_correct": []
            }
            
            for q_data in questions:
                question_id += 1
                print(f"  [{question_id}/50] {q_data['q'][:50]}...")
                
                # Get answer
                answer = agent.answer_question(q_data['q'])
                
                # Evaluate
                eval_result = self.judge.evaluate(q_data, answer)
                
                # Store
                result = {
                    "question_id": question_id,
                    "cluster": cluster_name,
                    "question": q_data['q'],
                    "answer": answer,
                    **eval_result
                }
                all_results.append(result)
                
                # Collect scores
                cluster_scores["correctness"].append(eval_result["correctness"])
                cluster_scores["completeness"].append(eval_result["completeness"])
                cluster_scores["faithfulness"].append(eval_result["faithfulness"])
                cluster_scores["handoff_correct"].append(1.0 if eval_result["handoff_correct"] else 0.0)
                
                print(f"      C:{eval_result['correctness']:.2f} | "
                      f"Comp:{eval_result['completeness']:.2f} | "
                      f"Faith:{eval_result['faithfulness']:.2f}")
                
                # LLMACE learning
                if llmace:
                    llmace.reflect(
                        query=q_data['q'],
                        response=answer,
                        success=(eval_result['correctness'] >= 0.7),
                        feedback=eval_result['reasoning'],
                        auto_update=True,
                        run_grow_and_refine=True
                    )
                    if question_id % 10 == 0:
                        print(f"      📚 Playbook: {len(llmace.context)} bullets")
            
            # Cluster metrics
            cluster_results[cluster_name] = ClusterMetrics(
                cluster_name=cluster_name,
                questions_count=len(questions),
                avg_correctness=statistics.mean(cluster_scores["correctness"]),
                avg_completeness=statistics.mean(cluster_scores["completeness"]),
                avg_faithfulness=statistics.mean(cluster_scores["faithfulness"]),
                handoff_accuracy=statistics.mean(cluster_scores["handoff_correct"])
            )
        
        # Calculate progression
        early_scores = [r["correctness"] for r in all_results[:20]]
        mid_scores = [r["correctness"] for r in all_results[20:35]]
        late_scores = [r["correctness"] for r in all_results[35:]]
        
        progression = ProgressionMetrics(
            early_correctness=statistics.mean(early_scores),
            mid_correctness=statistics.mean(mid_scores),
            late_correctness=statistics.mean(late_scores),
            improvement=statistics.mean(late_scores) - statistics.mean(early_scores)
        )
        
        return cluster_results, progression, all_results


def print_comparison(baseline_clusters, baseline_prog, llmace_clusters, llmace_prog):
    """Print detailed comparison."""
    
    print("\n" + "="*80)
    print("📊 LEARNING PROGRESSION COMPARISON")
    print("="*80)
    
    print(f"\n{'Stage':<20} {'Baseline':>15} {'LLMACE':>15} {'Improvement':>15}")
    print("-"*70)
    print(f"{'Early (Q1-20)':<20} {baseline_prog.early_correctness:>14.1%} {llmace_prog.early_correctness:>14.1%} {(llmace_prog.early_correctness - baseline_prog.early_correctness):>+14.1%}")
    print(f"{'Mid (Q21-35)':<20} {baseline_prog.mid_correctness:>14.1%} {llmace_prog.mid_correctness:>14.1%} {(llmace_prog.mid_correctness - baseline_prog.mid_correctness):>+14.1%}")
    print(f"{'Late (Q36-50)':<20} {baseline_prog.late_correctness:>14.1%} {llmace_prog.late_correctness:>14.1%} {(llmace_prog.late_correctness - baseline_prog.late_correctness):>+14.1%}")
    print("-"*70)
    print(f"{'Learning Gain':<20} {baseline_prog.improvement:>+14.1%} {llmace_prog.improvement:>+14.1%} {(llmace_prog.improvement - baseline_prog.improvement):>+14.1%}")
    
    print("\n" + "="*80)
    print("📂 CLUSTER PERFORMANCE")
    print("="*80)
    
    print(f"\n{'Cluster':<25} {'Metric':<15} {'Baseline':>12} {'LLMACE':>12} {'Δ':>10}")
    print("-"*80)
    
    for cluster_name in baseline_clusters.keys():
        b = baseline_clusters[cluster_name]
        l = llmace_clusters[cluster_name]
        
        print(f"{cluster_name:<25} {'Correctness':<15} {b.avg_correctness:>11.1%} {l.avg_correctness:>11.1%} {(l.avg_correctness - b.avg_correctness):>+9.1%}")
        print(f"{'':25} {'Faithfulness':<15} {b.avg_faithfulness:>11.1%} {l.avg_faithfulness:>11.1%} {(l.avg_faithfulness - b.avg_faithfulness):>+9.1%}")
        print(f"{'':25} {'Handoff Acc':<15} {b.handoff_accuracy:>11.1%} {l.handoff_accuracy:>11.1%} {(l.handoff_accuracy - b.handoff_accuracy):>+9.1%}")
        print()
    
    print("="*80)
    
    # Verdict
    if llmace_prog.improvement > baseline_prog.improvement + 0.05:
        print("🏆 LLMACE shows significantly better learning progression!")
    elif llmace_prog.improvement > baseline_prog.improvement:
        print("✅ LLMACE demonstrates improved learning over baseline")
    else:
        print("📊 Similar learning curves between methods")
    
    print("="*80)


def main():
    """Run FAQ benchmark."""
    
    # Load environment
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not openrouter_key and not openai_key:
        print("❌ No API keys found")
        return
    
    # Setup
    if openrouter_key:
        llm_client = OpenAI(api_key=openrouter_key, base_url="https://openrouter.ai/api/v1")
        embedding_client = OpenAI(api_key=openai_key) if openai_key else llm_client
        model = os.getenv("TEST_MODEL", "x-ai/grok-beta")
        judge_model = os.getenv("JUDGE_MODEL", "google/gemini-2.5-flash")
    else:
        llm_client = OpenAI(api_key=openai_key)
        embedding_client = llm_client
        model = judge_model = "google/gemini-2.5-flash"
    
    print("="*80)
    print("FAQ BENCHMARK - CloudSync Business Context Learning")
    print("="*80)
    print(f"Model: {model}")
    print(f"Judge: {judge_model}")
    print(f"Questions: 50 (10 semantic clusters)")
    
    # Run benchmarks
    benchmark = FAQBenchmark(llm_client, embedding_client, model, judge_model)
    
    baseline_clusters, baseline_prog, baseline_results = benchmark.run(use_llmace=False)
    llmace_clusters, llmace_prog, llmace_results = benchmark.run(use_llmace=True)
    
    # Compare
    print_comparison(baseline_clusters, baseline_prog, llmace_clusters, llmace_prog)
    
    # Save
    results = {
        "timestamp": time.time(),
        "model": model,
        "baseline": {
            "clusters": {k: v.to_dict() for k, v in baseline_clusters.items()},
            "progression": baseline_prog.to_dict(),
            "all_results": baseline_results
        },
        "llmace": {
            "clusters": {k: v.to_dict() for k, v in llmace_clusters.items()},
            "progression": llmace_prog.to_dict(),
            "all_results": llmace_results
        }
    }
    
    filename = f"faq_benchmark_{int(time.time())}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {filename}\n")


if __name__ == "__main__":
    main()
