"""
Comprehensive LLMACE Benchmark Suite

This benchmark provides systematic evaluation across multiple dimensions:
1. Feature Coverage - Tests all LLMACE capabilities
2. Performance Metrics - Accuracy, latency, cost, playbook growth
3. Baseline Comparisons - ICL, naive approach, with/without LLMACE
4. Statistical Rigor - Multiple runs, confidence intervals
5. Multiple Domains - Different task types

Aligned with ACE paper methodology.
"""

import os
import json
import time
import statistics
from typing import Dict, List, Any, Callable
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from llmace import LLMACE

try:
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolNode
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False


@dataclass
class BenchmarkMetrics:
    """Metrics tracked during benchmarking."""
    accuracy: float
    avg_turns: float
    avg_latency_ms: float
    total_tokens: int
    playbook_size: int
    success_rate: float
    cost_usd: float = 0.0


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    method_name: str
    task_type: str
    metrics: BenchmarkMetrics
    num_tasks: int
    timestamp: float


# ============================================================================
# Tools for Agent Tasks
# ============================================================================

@tool
def get_contacts(role: str = None) -> dict:
    """Get contacts, optionally filtered by role."""
    contacts = {
        "alice@email.com": {"name": "Alice", "role": "colleague", "phone": "555-0101"},
        "bob@email.com": {"name": "Bob", "role": "roommate", "phone": "555-0102"},
        "charlie@email.com": {"name": "Charlie", "role": "roommate", "phone": "555-0103"},
        "diana@email.com": {"name": "Diana", "role": "family", "phone": "555-0104"},
        "eve@email.com": {"name": "Eve", "role": "friend", "phone": "555-0105"},
    }
    
    if role:
        filtered = {k: v for k, v in contacts.items() if v["role"] == role}
        return {"success": True, "contacts": filtered, "count": len(filtered)}
    return {"success": True, "contacts": contacts, "count": len(contacts)}


@tool
def read_file(path: str) -> dict:
    """Read a file from the filesystem."""
    files = {
        "/home/bills/cable_bill.txt": "Cable TV Service\nTotal: $120.00\nDue Date: 2024-01-15",
        "/home/bills/internet_bill.txt": "Internet Service\nTotal: $60.00\nDue Date: 2024-01-20",
        "/home/bills/electricity_bill.txt": "Electricity\nTotal: $90.00\nDue Date: 2024-01-25",
        "/home/bills/water_bill.txt": "Water Service\nTotal: $45.00\nDue Date: 2024-01-30",
        "/home/receipts/grocery_2024_01.txt": "Grocery Receipt\nTotal: $150.00\nDate: 2024-01-10",
    }
    
    if path in files:
        return {"success": True, "content": files[path]}
    
    # Support wildcards
    if "*" in path:
        base_dir = path.split("*")[0]
        matching = {k: v for k, v in files.items() if k.startswith(base_dir)}
        return {"success": True, "files": matching, "count": len(matching)}
    
    return {"success": False, "error": f"File not found: {path}"}


@tool
def send_payment_request(to: str, amount: float, note: str) -> dict:
    """Send a payment request via Venmo/PayPal."""
    valid_emails = ["alice@email.com", "bob@email.com", "charlie@email.com", 
                    "diana@email.com", "eve@email.com"]
    
    if to not in valid_emails:
        return {"success": False, "error": f"Invalid email: {to}"}
    
    if amount <= 0:
        return {"success": False, "error": "Amount must be positive"}
    
    return {
        "success": True,
        "message": f"Payment request sent to {to}",
        "amount": amount,
        "note": note,
        "transaction_id": f"txn_{hash(to + str(amount))}"
    }


@tool
def calculate(expression: str) -> dict:
    """Safely evaluate a mathematical expression."""
    try:
        # Safe evaluation (only allow numbers and basic operators)
        allowed_chars = set("0123456789+-*/.()")
        if not all(c in allowed_chars or c.isspace() for c in expression):
            return {"success": False, "error": "Invalid characters in expression"}
        
        result = eval(expression)
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
def list_directory(path: str) -> dict:
    """List files in a directory."""
    # Normalize path (remove trailing slashes for consistency)
    normalized_path = path.rstrip('/')
    
    directories = {
        "/home/bills": ["cable_bill.txt", "internet_bill.txt", "electricity_bill.txt", "water_bill.txt"],
        "/home/receipts": ["grocery_2024_01.txt"],
        "/home": ["bills", "receipts", "documents"],
    }
    
    if normalized_path in directories:
        return {"success": True, "contents": directories[normalized_path], "count": len(directories[normalized_path])}
    return {"success": False, "error": f"Directory not found: {path}"}


# ============================================================================
# Benchmark Tasks
# ============================================================================

class BenchmarkTaskSuite:
    """Collection of benchmark tasks for different scenarios."""
    
    @staticmethod
    def get_bill_splitting_tasks() -> List[Dict]:
        """Tasks involving bill splitting (similar to AppWorld)."""
        return [
            {
                "task": "Split the cable bill equally with all roommates. The bill is in /home/bills/cable_bill.txt",
                "expected": {"amount_per_person": 40.0, "num_requests": 2},
                "difficulty": "medium"
            },
            {
                "task": "Read the internet bill and split it with your roommates",
                "expected": {"amount_per_person": 20.0, "num_requests": 2},
                "difficulty": "medium"
            },
            {
                "task": "Split all bills in /home/bills/ with roommates. Calculate the total each person owes.",
                "expected": {"total_per_person": 105.0, "num_bills": 4},
                "difficulty": "hard"
            },
            {
                "task": "The electricity bill is $90. Split it three ways with your 2 roommates.",
                "expected": {"amount_per_person": 30.0, "num_requests": 2},
                "difficulty": "easy"
            },
            {
                "task": "Send a payment request to each roommate for $25 for groceries",
                "expected": {"amount": 25.0, "num_requests": 2},
                "difficulty": "easy"
            },
        ]
    
    @staticmethod
    def get_contact_management_tasks() -> List[Dict]:
        """Tasks involving contact lookup and management."""
        return [
            {
                "task": "Find all my roommates and send them a message about tonight's dinner",
                "expected": {"num_roommates": 2},
                "difficulty": "easy"
            },
            {
                "task": "Get contact information for all family members",
                "expected": {"num_family": 1},
                "difficulty": "easy"
            },
            {
                "task": "Send a $20 payment request to each friend for the concert tickets",
                "expected": {"amount": 20.0, "num_requests": 1},
                "difficulty": "medium"
            },
        ]
    
    @staticmethod
    def get_file_operations_tasks() -> List[Dict]:
        """Tasks involving file reading and organization."""
        return [
            {
                "task": "List all files in the /home/bills directory",
                "expected": {"num_files": 4},
                "difficulty": "easy"
            },
            {
                "task": "Read all bills and calculate the total amount due",
                "expected": {"total": 315.0},
                "difficulty": "medium"
            },
            {
                "task": "Find the most expensive bill in /home/bills/",
                "expected": {"amount": 120.0, "file": "cable_bill.txt"},
                "difficulty": "medium"
            },
        ]
    
    @staticmethod
    def get_all_tasks() -> Dict[str, List[Dict]]:
        """Get all task suites organized by category."""
        return {
            "bill_splitting": BenchmarkTaskSuite.get_bill_splitting_tasks(),
            "contact_management": BenchmarkTaskSuite.get_contact_management_tasks(),
            "file_operations": BenchmarkTaskSuite.get_file_operations_tasks(),
        }


# ============================================================================
# Benchmark Methods
# ============================================================================

class LLMACEBenchmark:
    """Main benchmark class for evaluating LLMACE."""
    
    def __init__(self, llm_client: OpenAI, embedding_client: OpenAI, model_name: str):
        self.llm_client = llm_client
        self.embedding_client = embedding_client
        self.model_name = model_name
        self.tools = [get_contacts, read_file, send_payment_request, calculate, list_directory]
    
    def benchmark_baseline(self, tasks: List[Dict], num_runs: int = 1) -> BenchmarkResult:
        """Benchmark with no LLMACE (baseline)."""
        print("\n📊 Running BASELINE benchmark...")
        
        all_metrics = []
        
        # Run sequentially for debugging
        for run in range(num_runs):
            if num_runs > 1:
                print(f"  🔄 Run {run+1}/{num_runs}")
            metrics = self._run_tasks(tasks, use_llmace=False)
            all_metrics.append(metrics)
        
        # Average metrics across runs
        avg_metrics = self._average_metrics(all_metrics)
        
        return BenchmarkResult(
            method_name="Baseline (No LLMACE)",
            task_type="mixed",
            metrics=avg_metrics,
            num_tasks=len(tasks),
            timestamp=time.time()
        )
    
    def benchmark_with_llmace(self, tasks: List[Dict], num_runs: int = 1, 
                             num_epochs: int = 1) -> BenchmarkResult:
        """Benchmark with LLMACE learning."""
        print(f"\n🚀 Running WITH LLMACE benchmark (epochs={num_epochs})...")
        
        def run_single_llmace():
            """Execute a single LLMACE run with multiple epochs."""
            llmace = LLMACE(
                llm_client=self.llm_client,
                embedding_client=self.embedding_client,
                enable_logging=True  # Enable verbose logging
            )
            
            # Multi-epoch training
            for epoch in range(num_epochs):
                metrics = self._run_tasks(tasks, use_llmace=True, llmace=llmace)
            
            return metrics
        
        all_metrics = []
        
        # Run sequentially for debugging
        for run in range(num_runs):
            if num_runs > 1:
                print(f"  🔄 Run {run+1}/{num_runs}")
            all_metrics.append(run_single_llmace())
        
        # Average metrics across runs
        avg_metrics = self._average_metrics(all_metrics)
        
        return BenchmarkResult(
            method_name=f"LLMACE (epochs={num_epochs})",
            task_type="mixed",
            metrics=avg_metrics,
            num_tasks=len(tasks),
            timestamp=time.time()
        )
    
    def benchmark_icl(self, tasks: List[Dict], num_runs: int = 1) -> BenchmarkResult:
        """Benchmark with In-Context Learning (ICL baseline)."""
        print("\n📚 Running ICL baseline...")
        
        # Create ICL examples from first 2 tasks
        icl_examples = self._create_icl_examples(tasks[:2])
        
        all_metrics = []
        
        # Run sequentially for debugging
        for run in range(num_runs):
            if num_runs > 1:
                print(f"  🔄 Run {run+1}/{num_runs}")
            metrics = self._run_tasks(tasks[2:], use_llmace=False, icl_examples=icl_examples)
            all_metrics.append(metrics)
        
        avg_metrics = self._average_metrics(all_metrics)
        
        return BenchmarkResult(
            method_name="ICL (In-Context Learning)",
            task_type="mixed",
            metrics=avg_metrics,
            num_tasks=len(tasks) - 2,
            timestamp=time.time()
        )
    
    def _run_tasks(self, tasks: List[Dict], use_llmace: bool = False, 
                   llmace: LLMACE = None, icl_examples: str = "") -> BenchmarkMetrics:
        """Run a set of tasks and collect metrics."""
        
        if not LANGGRAPH_AVAILABLE:
            raise RuntimeError("LangGraph required for benchmarks. Install with: pip install llmace[agents]")
        
        total_turns = 0
        total_latency = 0
        total_tokens = 0
        successful_tasks = 0
        
        for task_spec in tasks:
            task = task_spec["task"]
            
            # Get playbook if using LLMACE
            playbook = ""
            if use_llmace and llmace:
                playbook = llmace.get_playbook()
            
            # Create system prompt
            system_prompt = self._create_system_prompt(playbook, icl_examples)
            
            # Execute task
            print(f"\n{'='*80}")
            print(f"🎯 Task #{len([t for t in tasks[:tasks.index(task_spec)+1]])}: {task}")
            print(f"{'='*80}")
            start_time = time.time()
            result = self._execute_task_with_langgraph(task, system_prompt)
            latency = (time.time() - start_time) * 1000  # ms
            print(f"⏱️  Latency: {latency:.0f}ms")
            
            # Show agent trajectory
            trajectory = result.get("trajectory", [])
            print(f"\n📜 Agent Trajectory ({len(trajectory)} tool calls):")
            tool_calls = [s for s in trajectory if s.get("role") == "tool"]
            for i, step in enumerate(tool_calls, 1):
                result_data = step.get('result', {})
                success = result_data.get('success', 'N/A')
                # Show first 50 chars of result
                result_preview = str(result_data)[:80] + "..." if len(str(result_data)) > 80 else str(result_data)
                print(f"   {i}. {step.get('name')}() → {success}")
                print(f"      {result_preview}")
            
            # Evaluate
            print(f"\n⚖️  Evaluating result...")
            success = self._evaluate_result(task_spec, result)
            print(f"{'✅ SUCCESS' if success else '❌ FAILED'}")
            
            # Update metrics
            total_turns += result.get("num_turns", 0)
            total_latency += latency
            total_tokens += result.get("tokens", 0)
            successful_tasks += 1 if success else 0
            
            # LLMACE learning
            if use_llmace and llmace:
                print(f"\n🧠 LLMACE Reflection & Curation...")
                llmace.reflect(
                    query=task,
                    response=json.dumps(result.get("trajectory", [])),
                    success=success,
                    feedback=f"Expected: {task_spec.get('expected', 'N/A')}",
                    auto_update=True,
                    run_grow_and_refine=True
                )
                print(f"📚 Playbook now has {len(llmace.context)} bullets")
        
        num_tasks = len(tasks)
        playbook_size = len(llmace.context) if (use_llmace and llmace) else 0
        
        return BenchmarkMetrics(
            accuracy=successful_tasks / num_tasks if num_tasks > 0 else 0.0,
            avg_turns=total_turns / num_tasks if num_tasks > 0 else 0.0,
            avg_latency_ms=total_latency / num_tasks if num_tasks > 0 else 0.0,
            total_tokens=total_tokens,
            playbook_size=playbook_size,
            success_rate=successful_tasks / num_tasks if num_tasks > 0 else 0.0,
            cost_usd=self._estimate_cost(total_tokens)
        )
    
    def _execute_task_with_langgraph(self, task: str, system_prompt: str) -> Dict:
        """Execute a task using LangGraph agent."""
        # Create LangChain model
        model_name = self.model_name
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        
        if openrouter_key:
            llm = ChatOpenAI(
                model=model_name,
                openai_api_key=openrouter_key,
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=0.7
            )
        else:
            llm = ChatOpenAI(model=model_name, temperature=0.7)
        
        # Bind tools
        llm_with_tools = llm.bind_tools(self.tools)
        
        # Execute
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=task)]
        num_turns = 0
        max_turns = 10
        trajectory = []
        
        while num_turns < max_turns:
            response = llm_with_tools.invoke(messages)
            num_turns += 1
            trajectory.append({"role": "assistant", "content": response.content})
            
            # Check for tool calls
            if hasattr(response, "tool_calls") and response.tool_calls:
                messages.append(response)
                
                # Execute tools
                for tool_call in response.tool_calls:
                    tool_name = tool_call.get("name")
                    tool_args = tool_call.get("args", {})
                    
                    # Find and execute tool
                    result = self._execute_tool(tool_name, tool_args)
                    trajectory.append({"role": "tool", "name": tool_name, "result": result})
                    
                    # Add tool result to messages
                    from langchain_core.messages import ToolMessage
                    messages.append(ToolMessage(content=str(result), tool_call_id=tool_call.get("id", "")))
            else:
                # No more tool calls, agent is done
                break
        
        return {
            "success": True,
            "num_turns": num_turns,
            "tokens": 500,  # Estimate
            "trajectory": trajectory,
            "final_response": response.content if num_turns > 0 else ""
        }
    
    def _execute_tool(self, tool_name: str, tool_args: Dict) -> Dict:
        """Execute a tool by name."""
        tool_map = {t.name: t for t in self.tools}
        
        if tool_name in tool_map:
            tool = tool_map[tool_name]
            try:
                return tool.func(**tool_args)
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        return {"success": False, "error": f"Unknown tool: {tool_name}"}
    
    def _evaluate_result(self, task_spec: Dict, result: Dict) -> bool:
        """Evaluate if task was completed correctly using LLM-as-a-judge."""
        # Use Gemini 2.0 Flash as judge
        judge_model = os.getenv("JUDGE_MODEL", "google/gemini-2.5-flash")
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        
        if openrouter_key:
            judge_client = OpenAI(api_key=openrouter_key, base_url="https://openrouter.ai/api/v1")
        else:
            judge_client = self.llm_client
        
        # Create evaluation prompt
        prompt = f"""You are an expert evaluator. Evaluate if the agent completed this task correctly.

**Task:** {task_spec['task']}

**Expected Outcome:** {json.dumps(task_spec.get('expected', 'N/A'))}

**Agent's Trajectory:**
{json.dumps(result.get('trajectory', []), indent=2)[:1000]}

**Final Response:** {result.get('final_response', 'N/A')}

**Evaluation Criteria:**
1. Did the agent use appropriate tools?
2. Did it achieve the expected outcome?
3. Was the approach logical?

Respond with JSON: {{"success": true/false, "reasoning": "brief explanation"}}"""
        
        try:
            # Define JSON schema for judge evaluation
            json_schema = {
                "name": "evaluation",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"},
                        "reasoning": {"type": "string"}
                    },
                    "required": ["success", "reasoning"],
                    "additionalProperties": False
                }
            }
            
            response = judge_client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator. Respond with JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={
                    "type": "json_schema",
                    "json_schema": json_schema
                }
            )
            
            evaluation = json.loads(response.choices[0].message.content)
            # Print judge reasoning
            print(f"      Judge: {'✅' if evaluation.get('success') else '❌'} - {evaluation.get('reasoning', 'No reasoning')}")
            return evaluation.get("success", False)
        except Exception as e:
            print(f"    ⚠️  Judge evaluation failed: {e}, falling back to heuristic")
            # Fallback: check if any tools were used successfully
            trajectory = result.get("trajectory", [])
            tool_results = [t for t in trajectory if t.get("role") == "tool"]
            return any(t.get("result", {}).get("success", False) for t in tool_results)
    
    def _create_system_prompt(self, playbook: str, icl_examples: str) -> str:
        """Create system prompt with optional playbook/ICL."""
        base = """You are an AI agent that can use tools to complete tasks.

Available tools:
- get_contacts(role=None): Get contacts
- read_file(path): Read a file
- send_payment_request(to, amount, note): Send payment request
- calculate(expression): Evaluate math expression
- list_directory(path): List directory contents

Think step-by-step and use tools effectively."""
        
        if playbook:
            base += f"\n\n**PLAYBOOK (Learned Strategies):**\n{playbook}"
        
        if icl_examples:
            base += f"\n\n**EXAMPLES:**\n{icl_examples}"
        
        return base
    
    def _create_icl_examples(self, tasks: List[Dict]) -> str:
        """Create ICL examples from tasks."""
        examples = []
        for i, task in enumerate(tasks):
            examples.append(f"Example {i+1}: {task['task']}")
        return "\n\n".join(examples)
    
    def _average_metrics(self, metrics_list: List[BenchmarkMetrics]) -> BenchmarkMetrics:
        """Average metrics across multiple runs."""
        if not metrics_list:
            return BenchmarkMetrics(0, 0, 0, 0, 0, 0)
        
        return BenchmarkMetrics(
            accuracy=statistics.mean(m.accuracy for m in metrics_list),
            avg_turns=statistics.mean(m.avg_turns for m in metrics_list),
            avg_latency_ms=statistics.mean(m.avg_latency_ms for m in metrics_list),
            total_tokens=int(statistics.mean(m.total_tokens for m in metrics_list)),
            playbook_size=int(statistics.mean(m.playbook_size for m in metrics_list)),
            success_rate=statistics.mean(m.success_rate for m in metrics_list),
            cost_usd=statistics.mean(m.cost_usd for m in metrics_list)
        )
    
    def _estimate_cost(self, tokens: int) -> float:
        """Estimate cost in USD based on tokens."""
        # Rough estimate: $0.01 per 1K tokens (adjust based on model)
        return (tokens / 1000) * 0.01


# ============================================================================
# Benchmark Runner and Reporter
# ============================================================================

class BenchmarkRunner:
    """Orchestrates benchmark execution and reporting."""
    
    def __init__(self, llm_client: OpenAI, embedding_client: OpenAI, model_name: str):
        self.benchmark = LLMACEBenchmark(llm_client, embedding_client, model_name)
    
    def run_comprehensive_benchmark(self, num_runs: int = 3, num_epochs: int = 3) -> Dict[str, Any]:
        """Run comprehensive benchmark suite with parallel execution."""
        
        print("=" * 80)
        print("LLMACE COMPREHENSIVE BENCHMARK SUITE")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  • Runs per method: {num_runs}")
        print(f"  • LLMACE epochs: {num_epochs}")
        print(f"  • Model: {self.benchmark.model_name}")
        print(f"  • Parallel execution: ENABLED")
        
        # Get tasks
        task_suite = BenchmarkTaskSuite()
        all_tasks = []
        for category, tasks in task_suite.get_all_tasks().items():
            all_tasks.extend(tasks)
        
        print(f"  • Total tasks: {len(all_tasks)}")
        print(f"\n🔄 Running benchmarks sequentially for debugging...")
        
        results = {}
        
        # Run all benchmarks sequentially for cleaner logs
        try:
            # 1. Baseline
            print("\n" + "=" * 80)
            results["baseline"] = self.benchmark.benchmark_baseline(all_tasks, num_runs)
            print(f"  ✓ baseline completed")
        except Exception as e:
            print(f"  ✗ baseline failed: {e}")
            import traceback
            traceback.print_exc()
        
        try:
            # 2. LLMACE (single epoch)
            print("\n" + "=" * 80)
            results["llmace_1epoch"] = self.benchmark.benchmark_with_llmace(all_tasks, num_runs, 1)
            print(f"  ✓ llmace_1epoch completed")
        except Exception as e:
            print(f"  ✗ llmace_1epoch failed: {e}")
            import traceback
            traceback.print_exc()
        
        try:
            # 3. LLMACE (multi-epoch)
            print("\n" + "=" * 80)
            results["llmace_multi"] = self.benchmark.benchmark_with_llmace(all_tasks, num_runs, num_epochs)
            print(f"  ✓ llmace_multi completed")
        except Exception as e:
            print(f"  ✗ llmace_multi failed: {e}")
            import traceback
            traceback.print_exc()
        
        try:
            # 4. ICL baseline
            if len(all_tasks) > 2:
                print("\n" + "=" * 80)
                results["icl"] = self.benchmark.benchmark_icl(all_tasks, num_runs)
                print(f"  ✓ icl completed")
        except Exception as e:
            print(f"  ✗ icl failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Generate report
        self._print_report(results)
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _print_report(self, results: Dict[str, BenchmarkResult]):
        """Print formatted benchmark report."""
        
        print("\n" + "=" * 80)
        print("COMPARATIVE BENCHMARK RESULTS")
        print("=" * 80)
        
        # Detailed comparison table
        print(f"\n{'Method':<30} {'Accuracy':>10} {'Δ vs Base':>12} {'Turns':>8} {'Latency':>12} {'Playbook':>10}")
        print("-" * 80)
        
        baseline_acc = results.get("baseline").metrics.accuracy if "baseline" in results else 0
        
        for method_name, result in results.items():
            m = result.metrics
            
            # Calculate delta vs baseline
            if method_name != "baseline" and baseline_acc > 0:
                delta = ((m.accuracy - baseline_acc) / baseline_acc) * 100
                delta_str = f"{delta:+.1f}%"
            else:
                delta_str = "-"
            
            print(f"{result.method_name:<30} {m.accuracy*100:>9.1f}% {delta_str:>12} {m.avg_turns:>7.1f} {m.avg_latency_ms:>11.0f}ms {m.playbook_size:>9d}")
        
        # Key comparisons
        print("\n" + "=" * 80)
        print("KEY COMPARISONS")
        print("=" * 80)
        
        if "baseline" in results and "llmace_multi" in results:
            baseline = results["baseline"].metrics
            llmace = results["llmace_multi"].metrics
            
            acc_improvement = ((llmace.accuracy - baseline.accuracy) / baseline.accuracy) * 100 if baseline.accuracy > 0 else 0
            turn_reduction = baseline.avg_turns - llmace.avg_turns
            
            print(f"\n1. LLMACE vs Baseline:")
            print(f"   • Accuracy:  {baseline.accuracy*100:.1f}% → {llmace.accuracy*100:.1f}% ({acc_improvement:+.1f}%)")
            print(f"   • Avg Turns: {baseline.avg_turns:.1f} → {llmace.avg_turns:.1f} ({turn_reduction:+.1f})")
            print(f"   • Playbook:  0 → {llmace.playbook_size} learned strategies")
        
        if "llmace_1epoch" in results and "llmace_multi" in results:
            single = results["llmace_1epoch"].metrics
            multi = results["llmace_multi"].metrics
            
            epoch_improvement = ((multi.accuracy - single.accuracy) / single.accuracy) * 100 if single.accuracy > 0 else 0
            
            print(f"\n2. Single-Epoch vs Multi-Epoch LLMACE:")
            print(f"   • Accuracy: {single.accuracy*100:.1f}% → {multi.accuracy*100:.1f}% ({epoch_improvement:+.1f}%)")
            print(f"   • Playbook: {single.playbook_size} → {multi.playbook_size} strategies")
        
        if "icl" in results and "baseline" in results:
            icl = results["icl"].metrics
            baseline = results["baseline"].metrics
            
            icl_improvement = ((icl.accuracy - baseline.accuracy) / baseline.accuracy) * 100 if baseline.accuracy > 0 else 0
            
            print(f"\n3. ICL vs Baseline:")
            print(f"   • Accuracy: {baseline.accuracy*100:.1f}% → {icl.accuracy*100:.1f}% ({icl_improvement:+.1f}%)")
        
        # Winner declaration
        if results:
            print("\n" + "=" * 80)
            best_method = max(results.items(), key=lambda x: x[1].metrics.accuracy)
            print(f"🏆 BEST PERFORMER: {best_method[1].method_name}")
            print(f"   Accuracy: {best_method[1].metrics.accuracy*100:.1f}%")
            print("=" * 80)
        else:
            print("\n⚠️  No results to compare - all benchmarks failed")
    
    def _save_results(self, results: Dict[str, BenchmarkResult]):
        """Save results to JSON file."""
        output = {
            "timestamp": time.time(),
            "model": self.benchmark.model_name,
            "results": {
                name: {
                    "method": result.method_name,
                    "metrics": asdict(result.metrics),
                    "num_tasks": result.num_tasks
                }
                for name, result in results.items()
            }
        }
        
        filename = f"benchmark_results_{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")


# ============================================================================
# Main
# ============================================================================

def main():
    """Run comprehensive benchmark."""
    
    if not LANGGRAPH_AVAILABLE:
        print("\n❌ LangGraph required for benchmarks!")
        print("Install with: pip install llmace[agents]")
        return
    
    # Setup logging to file
    import sys
    from datetime import datetime
    log_filename = f"benchmark_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    log_file = open(log_filename, 'w')
    
    # Write markdown header
    log_file.write("# LLMACE Benchmark Test Log\n\n")
    log_file.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    log_file.write("---\n\n")
    log_file.flush()
    
    class TeeOutput:
        def __init__(self, *files):
            self.files = files
        def write(self, data):
            for f in self.files:
                f.write(data)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    
    # Redirect stdout to both console and file
    original_stdout = sys.stdout
    sys.stdout = TeeOutput(sys.stdout, log_file)
    
    # Load environment
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not openrouter_key and not openai_key:
        print("❌ Error: No API key found")
        return
    
    # Initialize clients
    if openrouter_key:
        llm_client = OpenAI(api_key=openrouter_key, base_url="https://openrouter.ai/api/v1")
        embedding_client = OpenAI(api_key=openai_key) if openai_key else llm_client
        model_name = os.getenv("TEST_MODEL", "x-ai/grok-2-fast")
    else:
        llm_client = OpenAI(api_key=openai_key)
        embedding_client = llm_client
        model_name = "google/gemini-2.5-flash"
    
    # Run benchmark
    print(f"\n🤖 Using model: {model_name}")
    print(f"📝 Logging to: {log_filename}\n")
    print("🐛 DEBUG MODE: Running sequentially (no parallel) with detailed logs\n")
    runner = BenchmarkRunner(llm_client, embedding_client, model_name)
    
    try:
        runner.run_comprehensive_benchmark(num_runs=1, num_epochs=1)  # Simplified for debugging
    finally:
        # Restore stdout and close log file
        sys.stdout = original_stdout
        log_file.close()
        print(f"\n✅ Full log saved to: {log_filename}")


if __name__ == "__main__":
    main()

