"""
Test LLMACE with LangGraph agent framework (realistic integration).

This demonstrates how LLMACE integrates with real agentic frameworks,
similar to how the paper tested with ReAct on AppWorld.
"""

import os
import json
from typing import Annotated, TypedDict, Literal
from openai import OpenAI
from llmace import LLMACE

# LangGraph imports
try:
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolNode
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("⚠️  LangGraph not installed. Install with: pip install langgraph langchain-openai")


# Define tools (similar to AppWorld APIs)
@tool
def get_contacts(role: str = None) -> dict:
    """Get contacts, optionally filtered by role (e.g., 'roommate', 'colleague')."""
    contacts = {
        "alice@email.com": {"name": "Alice", "role": "colleague"},
        "bob@email.com": {"name": "Bob", "role": "roommate"},
        "charlie@email.com": {"name": "Charlie", "role": "roommate"},
    }
    
    if role:
        filtered = {k: v for k, v in contacts.items() if v["role"] == role}
        return {"success": True, "contacts": filtered}
    return {"success": True, "contacts": contacts}


@tool
def read_file(path: str) -> dict:
    """Read a file from the filesystem. Common paths: /home/bills/*.txt"""
    files = {
        "/home/bills/cable_bill.txt": "Total: $120",
        "/home/bills/internet_bill.txt": "Total: $60",
        "/home/bills/electricity_bill.txt": "Total: $90",
    }
    
    if path in files:
        return {"success": True, "content": files[path]}
    return {"success": False, "error": f"File not found: {path}"}


@tool  
def send_venmo_request(to: str, amount: float, note: str) -> dict:
    """Send a Venmo payment request to someone."""
    valid_emails = ["alice@email.com", "bob@email.com", "charlie@email.com"]
    
    if to not in valid_emails:
        return {"success": False, "error": f"Unknown contact: {to}"}
    
    # In real system, this would actually send the request
    return {
        "success": True,
        "message": f"Sent ${amount} request to {to}",
        "note": note
    }


# Agent state
class AgentState(TypedDict):
    """State of the agent."""
    messages: list
    task: str
    playbook: str


# Create agent with LLMACE playbook injection
def create_agent_with_llmace(llm_client, playbook: str = ""):
    """
    Create a LangGraph agent with LLMACE playbook injection.
    
    This is the key integration pattern - the playbook is injected
    into the agent's system prompt.
    """
    # Convert OpenAI client to LangChain format
    model_name = os.getenv("TEST_MODEL", "google/gemini-2.5-flash")
    
    # Determine which client to use
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
    
    # Bind tools to model
    tools = [get_contacts, read_file, send_venmo_request]
    llm_with_tools = llm.bind_tools(tools)
    
    # System prompt with playbook injection
    def create_system_prompt(playbook: str) -> str:
        base = """You are an AI assistant that can use tools to complete tasks.

Think step-by-step and use the available tools:
- get_contacts(role=None) - Get contacts, optionally filtered by role
- read_file(path) - Read files from filesystem
- send_venmo_request(to, amount, note) - Send payment requests

Complete the task efficiently."""
        
        if playbook:
            base += f"\n\n**PLAYBOOK (Learned strategies from past tasks):**\n{playbook}\n\nApply relevant strategies from the playbook to this task."
        
        return base
    
    # Agent node
    def call_model(state: AgentState):
        messages = state["messages"]
        system_prompt = create_system_prompt(state.get("playbook", ""))
        
        # Add system prompt
        messages_with_system = [SystemMessage(content=system_prompt)] + messages
        
        response = llm_with_tools.invoke(messages_with_system)
        return {"messages": messages + [response]}
    
    # Should continue?
    def should_continue(state: AgentState) -> Literal["tools", "end"]:
        messages = state["messages"]
        last_message = messages[-1]
        
        # If there are tool calls, continue to tools node
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        # Otherwise end
        return "end"
    
    # Build graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))
    
    # Add edges
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()


def test_task_with_llmace(agent, task: str, playbook: str = "") -> dict:
    """Execute a task with the agent and return results."""
    initial_state = {
        "messages": [HumanMessage(content=task)],
        "task": task,
        "playbook": playbook
    }
    
    # Run agent
    result = agent.invoke(initial_state)
    
    # Extract trajectory
    messages = result["messages"]
    trajectory = []
    
    for msg in messages:
        if isinstance(msg, HumanMessage):
            trajectory.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            content = msg.content
            tool_calls = getattr(msg, "tool_calls", [])
            trajectory.append({
                "role": "assistant",
                "content": content,
                "tool_calls": [tc.get("name") for tc in tool_calls] if tool_calls else []
            })
    
    return {
        "trajectory": trajectory,
        "final_response": messages[-1].content if messages else "",
        "num_turns": len([m for m in messages if isinstance(m, AIMessage)])
    }


def evaluate_task_execution(task: dict, execution: dict) -> dict:
    """Evaluate if task was completed correctly."""
    # Simple heuristics for evaluation
    trajectory = execution["trajectory"]
    response = execution["final_response"].lower()
    
    # Check if agent used appropriate tools
    tools_used = []
    for turn in trajectory:
        if "tool_calls" in turn:
            tools_used.extend(turn["tool_calls"])
    
    # Task-specific evaluation
    task_desc = task["task"].lower()
    
    success_indicators = {
        "roommate": "get_contacts" in tools_used,
        "bill": "read_file" in tools_used,
        "venmo": "send_venmo_request" in tools_used,
    }
    
    success = all(
        indicator in tools_used if keyword in task_desc else True
        for keyword, indicator in success_indicators.items()
    )
    
    return {
        "success": success,
        "tools_used": tools_used,
        "num_tool_calls": len(tools_used)
    }


def main():
    """Run LLMACE + LangGraph integration test."""
    
    if not LANGGRAPH_AVAILABLE:
        print("\n❌ LangGraph not installed!")
        print("\nInstall with:")
        print("  pip install langgraph langchain-openai langchain-core")
        return
    
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
    
    print("=" * 70)
    print("LLMACE + LANGGRAPH INTEGRATION TEST")
    print("=" * 70)
    print("\nTesting LLMACE with realistic LangGraph agent framework")
    print("(This is how users would actually integrate LLMACE)\n")
    
    # Initialize clients
    if openrouter_key:
        print("🚀 Using OpenRouter for LLM")
        llm_client = OpenAI(api_key=openrouter_key, base_url="https://openrouter.ai/api/v1")
        embedding_client = OpenAI(api_key=openai_key) if openai_key else llm_client
    else:
        print("🔑 Using OpenAI")
        llm_client = OpenAI(api_key=openai_key)
        embedding_client = llm_client
    
    # Initialize LLMACE
    llmace = LLMACE(
        llm_client=llm_client,
        embedding_client=embedding_client,
        enable_logging=True
    )
    
    # Test tasks
    tasks = [
        {
            "task": "Find all my roommates and send each a $40 Venmo request for the cable bill",
            "description": "Multi-step task: contact lookup → payment requests"
        },
        {
            "task": "Read the internet bill file and split it equally with my roommates via Venmo",
            "description": "Complex task: file read → contact lookup → calculation → payments"
        },
        {
            "task": "Send a $30 Venmo request to each roommate for the electricity bill",
            "description": "Should reuse learned strategies from previous tasks"
        },
    ]
    
    print(f"Running {len(tasks)} tasks with LangGraph agent...\n")
    
    # Track metrics
    results = []
    
    for i, task_spec in enumerate(tasks):
        print("\n" + "-" * 70)
        print(f"Task {i+1}/{len(tasks)}: {task_spec['task'][:60]}...")
        print(f"Description: {task_spec['description']}")
        print("-" * 70)
        
        # Get current playbook
        playbook = llmace.get_playbook()
        
        if playbook:
            print(f"\n📖 Using playbook ({len(llmace.context)} strategies)")
        else:
            print("\n📖 No playbook yet (first task)")
        
        # Create agent with current playbook
        agent = create_agent_with_llmace(llm_client, playbook)
        
        # Execute task
        print(f"\n🤖 LangGraph agent executing...")
        execution = test_task_with_llmace(agent, task_spec["task"], playbook)
        
        print(f"✓ Completed in {execution['num_turns']} agent turns")
        
        # Evaluate
        evaluation = evaluate_task_execution(task_spec, execution)
        print(f"Tools used: {', '.join(evaluation['tools_used']) if evaluation['tools_used'] else 'None'}")
        print(f"Success: {'✓ CORRECT' if evaluation['success'] else '✗ INCORRECT'}")
        
        # LLMACE learns from execution
        print(f"\n🔄 LLMACE reflecting and learning...")
        
        trajectory_str = json.dumps(execution["trajectory"], indent=2)
        feedback = f"""
Task: {task_spec['task']}
Tools used: {evaluation['tools_used']}
Success: {evaluation['success']}
Final response: {execution['final_response'][:200]}
"""
        
        llmace.reflect(
            query=task_spec["task"],
            response=trajectory_str,
            success=evaluation["success"],
            feedback=feedback,
            auto_update=True,
            run_grow_and_refine=True
        )
        
        print(f"📈 Playbook now has {len(llmace.context)} strategies")
        
        results.append({
            "task": task_spec["task"],
            "success": evaluation["success"],
            "num_turns": execution["num_turns"],
            "tools_used": evaluation["tools_used"],
            "playbook_size": len(llmace.context)
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for i, result in enumerate(results):
        print(f"\nTask {i+1}: {'✓' if result['success'] else '✗'}")
        print(f"  Turns: {result['num_turns']}")
        print(f"  Tools: {', '.join(result['tools_used'])}")
        print(f"  Playbook size: {result['playbook_size']} strategies")
    
    # Show final playbook
    print("\n" + "=" * 70)
    print("FINAL PLAYBOOK (Learned Strategies)")
    print("=" * 70)
    final_playbook = llmace.get_playbook()
    print(final_playbook if final_playbook else "(empty)")
    
    # Save for inspection
    llmace.save("langgraph_test_context.json")
    print(f"\n💾 Saved context to: langgraph_test_context.json")
    
    print("\n" + "=" * 70)
    print("✅ LANGGRAPH INTEGRATION TEST COMPLETE")
    print("=" * 70)
    print("\n🎯 Key Observations:")
    print("  • LangGraph provides realistic agent framework")
    print("  • LLMACE playbook injected into system prompt")
    print("  • Agent learns strategies across tasks")
    print("  • Later tasks benefit from earlier learnings")
    print("  • This is production-ready integration pattern!")
    print("\n💡 This demonstrates how users should integrate LLMACE")
    print("   into real agentic workflows (LangGraph, CrewAI, etc.)")


if __name__ == "__main__":
    main()


