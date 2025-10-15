# Testing LLMACE Effectiveness

This guide shows you how to verify that LLMACE actually improves LLM performance using rigorous testing methodologies.

## ⚠️ Important: Types of Tests

Based on the ACE paper, there are three types of tests:

### 1. **Simple Chat Flows** (`test_llmace_effectiveness.py`)
- ❌ **Not ideal for ACE** - too simple!
- Single-turn Q&A tasks (math, conversions)
- No tool usage, no environment feedback
- **Use for**: Quick sanity checks only

### 2. **LangGraph Integration** (`test_langgraph_integration.py`) ⭐ **RECOMMENDED**
- ✅ **Best approach** - realistic agent framework
- Uses industry-standard LangGraph framework
- Multi-turn reasoning with real tool calling
- Shows production-ready integration pattern
- **Use for**: Realistic evaluation + integration demo

### 3. **Custom Agentic Flows** (`test_agentic_flows.py`)
- ✅ Good for understanding internals
- Custom mock environment
- Educational but not production-ready
- **Use for**: Learning how ACE works

**Paper's approach**: AppWorld benchmark with ReAct agent framework, similar to our LangGraph test.

## Quick Verification Test

For a quick sanity check that LLMACE is working:

### Option 1: Using Environment Variables
```bash
cd tests

# Recommended: OpenRouter for LLM + OpenAI for embeddings
export OPENROUTER_API_KEY='sk-or-v1-your-openrouter-key'
export OPENAI_API_KEY='sk-your-openai-key'  # For embeddings (best quality)

# OR OpenRouter only (will use for both LLM and embeddings)
export OPENROUTER_API_KEY='sk-or-v1-your-openrouter-key'

# OR OpenAI only (will use for both LLM and embeddings)
export OPENAI_API_KEY='sk-your-openai-key'

python quick_test.py
```

**Priority Logic:**
- If `OPENROUTER_API_KEY` exists → Use OpenRouter for LLM
- If `OPENAI_API_KEY` exists → Use OpenAI for embeddings (recommended)
- Falls back intelligently if only one key present

### Option 2: Using .env File (Recommended)
```bash
# 1. Copy the example env file
cp env.example .env

# 2. Edit .env with your API keys
nano .env  # or use your favorite editor

# 3. Run the test
cd tests
python quick_test.py
```

This will:
- Initialize LLMACE
- Run 2 reflection cycles
- Show playbook evolution
- Test serialization

Expected output: `✅ ALL TESTS PASSED!`

---

## LangGraph Integration Test ⭐ **RECOMMENDED**

**The best way to test LLMACE** - uses real agent framework!

### Installation

```bash
pip install llmace[agents]
# Or manually:
pip install langgraph langchain-openai langchain-core
```

### Run Test

```bash
cd tests
python test_langgraph_integration.py
```

### What This Tests

**Production-ready integration** with LangGraph:
- ✅ Industry-standard agent framework (not a toy)
- ✅ Real tool calling pattern
- ✅ State management across turns
- ✅ Playbook injection into system prompt
- ✅ Shows how users actually integrate LLMACE

**Why LangGraph?**
- Most popular Python agent framework
- Used by thousands of production applications
- Clean tool definition and execution
- Proper state management
- **This is how you'd actually use LLMACE in production**

### Example Integration Pattern

```python
from langgraph.graph import StateGraph
from llmace import LLMACE

# Initialize LLMACE
llmace = LLMACE(llm_client=client, embedding_client=embedding_client)

# Get evolved playbook
playbook = llmace.get_playbook()

# Inject into agent's system prompt
system_prompt = f"""You are an AI agent.

**PLAYBOOK (Learned Strategies):**
{playbook}

Use relevant strategies from the playbook."""

# Create LangGraph agent with playbook
agent = create_agent(system_prompt)

# After execution, reflect and learn
llmace.reflect(
    query=task,
    response=execution_result,
    auto_update=True
)
```

### Expected Output

```
LLMACE + LANGGRAPH INTEGRATION TEST
======================================================================

Task 1: Find roommates and send $40 Venmo requests
📖 No playbook yet (first task)
🤖 LangGraph agent executing...
✓ Completed in 3 agent turns
Tools used: get_contacts, send_venmo_request
📈 Playbook now has 2 strategies

Task 2: Read internet bill and split with roommates
📖 Using playbook (2 strategies)
🤖 LangGraph agent executing...
✓ Completed in 4 agent turns
Tools used: read_file, get_contacts, send_venmo_request
📈 Playbook now has 4 strategies

FINAL PLAYBOOK:
• Use get_contacts(role='roommate') to find roommates reliably
• Read bills from /home/bills/ directory  
• Calculate splits including yourself in the count
• ... more learned strategies ...

✅ LANGGRAPH INTEGRATION TEST COMPLETE

💡 This demonstrates production-ready LLMACE integration!
```

### Why This Is the Best Test

| Aspect | LangGraph Test | Simple Q&A | Custom Mock |
|--------|---------------|------------|-------------|
| Framework | ✅ Real (LangGraph) | ❌ None | ❌ Toy mock |
| Tool calling | ✅ Production pattern | ❌ None | ⚠️ Simplified |
| State management | ✅ Built-in | ❌ None | ⚠️ Manual |
| Integration demo | ✅ Yes | ❌ No | ❌ No |
| User relevance | ✅ High | ❌ Low | ⚠️ Medium |

---

## Agentic Flow Test (Educational)

**This is how ACE should be tested** according to the paper!

```bash
cd tests
python test_agentic_flows.py
```

### What This Tests

**Multi-turn agent tasks** similar to AppWorld benchmark:
- Agent uses tools/APIs across multiple turns
- Environment provides execution feedback
- Playbook accumulates actionable strategies
- Later tasks benefit from earlier learnings

**Example task**: "Split the cable bill with all my roommates"
1. Agent calls `get_contacts(role='roommate')` → finds Bob, Charlie
2. Agent calls `read_file('/home/bills/cable_bill.txt')` → gets $120
3. Agent calculates: $120 / 3 people = $40 each
4. Agent calls `send_venmo_request(bob, 40)` and `send_venmo_request(charlie, 40)`
5. LLMACE reflects: learns "Always use get_contacts(role='roommate') to find roommates"
6. Next task uses this strategy automatically!

### Why This Is Better

| Simple Chat Test | Agentic Flow Test |
|-----------------|-------------------|
| ❌ Single-turn Q&A | ✅ Multi-turn reasoning |
| ❌ No tool usage | ✅ Tool/API calls |
| ❌ No feedback loop | ✅ Execution feedback |
| ❌ Trivial learning | ✅ Complex strategies |
| Use for: Sanity check | Use for: Real evaluation |

### Expected Output

```
Task 1: Split cable bill with roommates
📖 No playbook yet (first task)
🤖 Agent executing task...
✓ Completed in 5 turns
Success: True
📈 Playbook now has 3 strategies

Task 2: Send grocery requests to roommates  
📖 Using playbook (3 strategies)
🤖 Agent executing task...
✓ Completed in 3 turns (faster! using learned strategies)
Success: True
📈 Playbook now has 5 strategies

FINAL PLAYBOOK:
• Always use get_contacts(role='roommate') to find roommates
• Bills are typically in /home/bills/ directory
• Split costs by dividing total by (num_roommates + 1) including yourself
• ... more strategies ...
```

---

## Comprehensive Effectiveness Test (Simple Q&A)

To rigorously test if LLMACE improves LLM performance using **LLM-as-a-judge** evaluation:

### Option 1: Using .env File (Recommended)
```bash
# 1. Copy and configure .env
cp env.example .env
nano .env  # Add your API keys

# 2. Run the test
cd tests
python test_llmace_effectiveness.py
```

### Option 2: Environment Variables
```bash
cd tests

# For OpenRouter (recommended)
export OPENROUTER_API_KEY='sk-or-v1-your-key'
export TEST_MODELS='x-ai/grok-2-fast,openai/gpt-4.5-turbo'  # Multiple models (comma-separated)
export JUDGE_MODEL='google/gemini-2.0-flash-exp:free'       # Gemini 2.0 Flash for evaluation

# OR for OpenAI directly
export OPENAI_API_KEY='sk-your-key'

python test_llmace_effectiveness.py
```

### Why Use OpenRouter?
- ✅ Access to 100+ models from one API
- ✅ Use Claude, GPT-4, Llama, and more
- ✅ Automatic fallback and rate limiting
- ✅ Competitive pricing
- ✅ Get your key at: https://openrouter.ai/keys

### Multi-Model Testing

The test supports evaluating LLMACE with **multiple models simultaneously**. This is useful for:
- Comparing LLMACE effectiveness across different model families
- Testing with both fast models (Grok) and powerful models (GPT-5)
- Ensuring your playbook generalizes across architectures

**Example `.env` configuration:**
```bash
# Test with both Grok (fast) and GPT-5 (powerful)
TEST_MODELS=x-ai/grok-2-fast,openai/gpt-4.5-turbo

# Use Gemini 2.0 Flash as judge (fast, accurate, cost-effective)
JUDGE_MODEL=google/gemini-2.0-flash-exp:free
```

**Output will show:**
```
🔄 Multi-model testing enabled: 2 models
   Models: x-ai/grok-2-fast, openai/gpt-4.5-turbo

TESTING WITH MODEL: x-ai/grok-2-fast
... results ...

TESTING WITH MODEL: openai/gpt-4.5-turbo
... results ...

MULTI-MODEL COMPARISON SUMMARY
📊 x-ai/grok-2-fast:
   Improvement: 18.1%
   LLMACE Avg Score: 8.5/10

📊 openai/gpt-4.5-turbo:
   Improvement: 15.3%
   LLMACE Avg Score: 9.2/10
```

### What This Test Does

The test runs in two phases:

**Phase 1: Baseline (No LLMACE)**
- Runs 8 test tasks without LLMACE
- Pure LLM responses with no evolving context
- Measures accuracy and quality scores

**Phase 2: With LLMACE (Learning Enabled)**
- Runs the same 8 tasks with LLMACE
- Each task adds to the playbook
- Later tasks benefit from earlier learnings
- Measures improvement over baseline

**Evaluation Method: LLM-as-a-Judge**
- GPT-4 acts as an expert evaluator
- Scores responses on 0-10 scale
- Evaluates correctness, completeness, clarity
- Provides detailed reasoning

### Expected Results

LLMACE should show:
- **Higher average scores** (typically +0.5 to +2.0 points)
- **Better accuracy** (more correct answers)
- **More consistency** (smaller score variance)
- **Progressive improvement** (later tasks perform better)

Example output:
```
📊 BASELINE (No LLMACE):
  Average Score: 7.2/10
  Accuracy: 75.0% (6 correct)

🚀 WITH LLMACE:
  Average Score: 8.5/10
  Accuracy: 87.5% (7 correct)

📈 IMPROVEMENT:
  Score Delta: +1.3 points
  Accuracy Delta: +12.5%
  Percentage Improvement: 18.1%

✅ LLMACE shows positive improvement!
```

---

## Custom Testing for Your Domain

To test LLMACE on your specific use case:

### 1. Create Your Test Tasks

```python
from tests.test_llmace_effectiveness import LLMACEEffectivenessTest, LLMJudge
from openai import OpenAI

# Define your domain-specific tasks
my_tasks = [
    {
        "query": "Your task 1",
        "ground_truth": "Expected answer (optional)"
    },
    {
        "query": "Your task 2",
        "ground_truth": "Expected answer (optional)"
    },
    # Add 5-10 tasks
]

# Run test
client = OpenAI(api_key="your-key")
tester = LLMACEEffectivenessTest(client)

baseline = tester.run_baseline(my_tasks)
llmace = tester.run_with_llmace(my_tasks)
comparison = tester.compare_results(baseline, llmace)
tester.print_report(comparison)
```

### 2. Design Good Test Tasks

Good test tasks should:
- ✅ Be **solvable** by the LLM (not impossible)
- ✅ Have **some difficulty** (not trivial)
- ✅ Be **similar in structure** (so learning transfers)
- ✅ Have **clear success criteria** (for evaluation)
- ✅ Represent **real-world use cases**

Example domains:
- **Math problems**: Calculations, conversions, word problems
- **Code generation**: Similar programming tasks
- **Data analysis**: Query interpretation, insights extraction
- **Creative writing**: Similar genres/styles
- **Customer support**: Common issue types

### 3. Interpret Results

**Positive indicators:**
- Average score increases by ≥0.5 points
- Accuracy improves by ≥10%
- Later tasks show better scores than earlier ones
- Playbook grows with relevant insights

**When LLMACE might not help:**
- Tasks are too diverse (no shared learning)
- Tasks are trivial (already 100% accuracy)
- Tasks are impossible (LLM can't solve them)
- Not enough iterations (need 5+ tasks)

---

## A/B Testing in Production

For production validation:

```python
from llmace import LLMACE
from openai import OpenAI
import random

client = OpenAI(api_key="your-key")

# Group A: With LLMACE
llmace = LLMACE(llm_client=client, embedding_client=client)

# Group B: Without LLMACE (baseline)
# Just use client directly

def handle_request(user_query: str):
    # Randomly assign to A or B
    use_llmace = random.random() < 0.5
    
    if use_llmace:
        playbook = llmace.get_playbook()
        system_prompt = f"You are helpful.\n\n{playbook}"
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]
        )
        
        answer = response.choices[0].message.content
        
        # Learn from this interaction
        llmace.reflect(
            query=user_query,
            response=answer,
            auto_update=True
        )
        
        return answer, "llmace"
    else:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": user_query}
            ]
        )
        return response.choices[0].message.content, "baseline"

# Log results and track metrics:
# - User satisfaction ratings
# - Task success rates
# - Response quality scores
```

Track metrics:
- User satisfaction (ratings, thumbs up/down)
- Task completion rates
- Error rates
- Response time
- Cost per query

---

## Metrics to Track

### Quantitative Metrics
- **Average Score**: Mean LLM-as-a-judge score (0-10)
- **Accuracy**: Percentage of correct answers
- **Consistency**: Standard deviation of scores
- **Improvement Rate**: Score improvement per iteration

### Qualitative Metrics
- **Relevance**: Are playbook items relevant to tasks?
- **Diversity**: Does playbook cover different aspects?
- **Clarity**: Are insights clear and actionable?
- **Transfer**: Do insights help with new tasks?

### System Metrics
- **Playbook Size**: Number of bullets over time
- **Deduplication Rate**: Redundant bullets removed
- **Reflection Time**: Time to generate insights
- **Context Size**: Tokens used by playbook

---

## Troubleshooting

### "No improvement detected"

Possible causes:
1. **Tasks too diverse**: No shared learning between tasks
   - Fix: Use more similar tasks in same domain
2. **Too few iterations**: Need more data to learn
   - Fix: Run 10+ tasks instead of 5
3. **Tasks too easy**: Already at ceiling performance
   - Fix: Use harder tasks
4. **Reflection disabled**: Auto-update not enabled
   - Fix: Check `auto_update=True` in `reflect()`

### "Playbook not growing"

Check:
1. Is `auto_update=True` in `reflect()`?
2. Is LLM client properly initialized?
3. Are reflections generating insights? (enable logging)
4. Is deduplication too aggressive? (check threshold)

### "Inconsistent results"

Solutions:
1. Lower temperature (0.3-0.5) for more deterministic results
2. Use more test tasks (10+ instead of 5)
3. Run test multiple times and average results
4. Use GPT-4 instead of GPT-3.5 for evaluation

---

## Advanced: Custom Evaluation

For domain-specific evaluation, create custom judge:

```python
class CustomJudge:
    def __init__(self, client: OpenAI):
        self.client = client
    
    def evaluate(self, query: str, response: str) -> Dict:
        # Your custom evaluation logic
        prompt = f"""
        Evaluate this response for [YOUR DOMAIN]:
        
        Query: {query}
        Response: {response}
        
        Rate on:
        1. Domain-specific criterion 1
        2. Domain-specific criterion 2
        ...
        
        Return JSON with score and reasoning.
        """
        
        result = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        return json.loads(result.choices[0].message.content)
```

---

## Best Practices

1. **Start Small**: Test with 5-10 tasks first
2. **Use Ground Truth**: Provide expected answers when possible
3. **Domain-Specific**: Test on your actual use case
4. **Multiple Runs**: Run tests 3-5 times for statistical significance
5. **Track Over Time**: Monitor metrics in production
6. **A/B Test**: Compare with and without LLMACE in real usage
7. **Iterate**: Adjust prompts and config based on results

---

## Example Test Results

Here's what a successful test looks like:

```
LLMACE EFFECTIVENESS TEST RESULTS
======================================================================

📊 BASELINE (No LLMACE):
  Average Score: 6.8/10
  Accuracy: 62.5% (5 correct)
  Score Range: 4-9

🚀 WITH LLMACE:
  Average Score: 8.3/10
  Accuracy: 87.5% (7 correct)
  Score Range: 7-10

📈 IMPROVEMENT:
  Score Delta: +1.5 points
  Accuracy Delta: +25.0%
  Percentage Improvement: 22.1%

✅ LLMACE shows positive improvement!
======================================================================
```

This demonstrates:
- ✅ 22% overall performance improvement
- ✅ 25% increase in correct answers
- ✅ More consistent scores (narrower range)
- ✅ Clear benefit from evolving context

---

## Next Steps

After validating LLMACE effectiveness:

1. **Tune Configuration**: Adjust dedup threshold, max bullets, etc.
2. **Customize Prompts**: Tailor reflection/curation prompts to your domain
3. **Deploy Gradually**: Start with A/B test in production
4. **Monitor Metrics**: Track real-world performance
5. **Iterate**: Continuously improve based on results

For questions or to share your results, see [CONTRIBUTING.md](CONTRIBUTING.md).

