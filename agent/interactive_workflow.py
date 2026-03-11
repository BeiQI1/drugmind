import os
import sys
import json
import time
import uuid
import datetime
import logging
from dotenv import load_dotenv
from typing import Dict, Any, Literal

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

from agent.state import AgentState
from agent.IntentAgent import intent_agent_node
from agent.TargetAgent import target_agent_node
from agent.GeneratorAgent import generator_agent_node
from agent.EvaluatorAgent import evaluator_agent_node
from agent.SynthesisAgent import synthesis_agent_node
from agent.ReportAgent import report_agent_node
from agent.CoordinatorAgent import coordinator_agent_node
from agent.RAGAgent import RAGAgent

# Load env vars
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    load_dotenv(env_path, override=True)

# --- Routing Logic ---

def route_after_intent(state: AgentState) -> Literal["coordinator_agent", END]:
    intent = state.get("intent")
    if intent in ["generation", "report", "synthesis_planning", "optimization", "evaluation"]:
        return "coordinator_agent"
    elif intent == "clarification_needed":
        return END
    else:
        # If intent is valid, try coordinator
        if intent and intent != "error":
            return "coordinator_agent"
        return END

def plan_router(state: AgentState) -> str:
    """
    Dynamic router that follows the plan in state.
    """
    plan = state.get("plan", [])
    step = state.get("plan_step", 0)
    
    print(f"\n[Router] Checking Plan Step {step}/{len(plan)}: {plan}")
    
    if not plan or step >= len(plan):
        print("[Router] Plan complete or empty. Ending workflow.")
        return END
        
    next_agent_name = plan[step]

    # CRITICAL FIX for Benchmark/API mode:
    # If the plan explicitly excludes ReportAgent (e.g. for benchmarking), 
    # but the logic somehow routed here, we must Ensure we DO NOT route to ReportAgent.
    # We check the 'planning_rules' or 'restrictions' in task_params if available.
    task_params = state.get("task_params", {})
    restrictions = task_params.get("restrictions", [])
    if "No PDF report" in restrictions and next_agent_name == "ReportAgent":
        print(f"[Router] ReportAgent blocked by restrictions. Ending workflow.")
        return END
    
    print(f"[Router] Routing to: {next_agent_name}")
    
    mapping = {
        "TargetAgent": "target_agent",
        "GeneratorAgent": "generator_agent",
        "EvaluatorAgent": "evaluator_agent",
        "SynthesisAgent": "synthesis_agent",
        "ReportAgent": "report_agent"
    }
    
    return mapping.get(next_agent_name, END)

# --- Node Wrappers with State Updates ---
# We wrap the existing nodes to ensure they update the loop counter correctly and provide CLI feedback

def intent_node_wrapper(state: AgentState) -> Dict[str, Any]:
    print("\n[IntentAgent] Thinking...")
    if not state.get("task_params"):
        state["task_params"] = {}
    if "intent_turn_id" not in state["task_params"]:
        state["task_params"]["intent_turn_id"] = str(state.get("run_id") or uuid.uuid4())[:8]
    result = intent_agent_node(state)
    
    intent = result.get("intent")
    response = result.get("results", {}).get("response", "")
    
    print(f"[IntentAgent] Intent: {intent}")
    if response:
        print(f"[IntentAgent] Response: {response}")
        
    if intent == "error":
        print(f"[IntentAgent] Error: {result.get('error')}")
    
    # Update messages history for LangGraph state
    user_msg = HumanMessage(content=state["user_input"])
    ai_msg = AIMessage(content=response if response else str(result))
    result["messages"] = [user_msg, ai_msg]
    
    # Initialize agent_logs in task_params
    params = result.get("task_params", {})
    logs = f"**Intent Agent**:\n- User Input: {state['user_input']}\n- Intent: {intent}\n- Response: {response}\n- Extracted Params: {json.dumps(params, indent=2)}"
    params["agent_logs"] = logs
    result["task_params"] = params
        
    return result

def target_node_wrapper(state: AgentState) -> Dict[str, Any]:
    print("\n[TargetAgent] Planning and Executing...")
    result = target_agent_node(state)
    
    if result.get("error"):
        print(f"[TargetAgent] Error: {result.get('error')}")
    else:
        target_results = result.get("results", {}).get("target_preparation", {})
        print("[TargetAgent] Execution Complete.")
        
        # Append to agent_logs
        current_params = state.get("task_params", {}).copy()
        current_logs = current_params.get("agent_logs", "")
        new_log = f"\n\n- **Target Agent**:\n  - Target Protein: {target_results.get('target_name', 'Unknown')}\n  - PDB ID: {target_results.get('pdb_id', 'N/A')}\n  - Pocket Center: {target_results.get('pocket_center', 'Auto-detected')}\n  - Full Results: {json.dumps(target_results, indent=2)}"
        current_params["agent_logs"] = current_logs + new_log
        
        # Merge updates
        result["task_params"] = current_params
        
    # Increment plan step
    result["plan_step"] = state.get("plan_step", 0) + 1
        
    return result

def generator_node_wrapper(state: AgentState) -> Dict[str, Any]:
    # Increment loop count before running generator
    current_loop = state.get("loop_count", 0) + 1
    state["loop_count"] = current_loop
    print(f"\n[GeneratorAgent] Starting Loop {current_loop}...")
    
    result = generator_agent_node(state)
    
    if result.get("error"):
        print(f"[GeneratorAgent] Error: {result.get('error')}")
    else:
        gen_results = result.get("results", {}).get("generation", {})
        # Calculate total count from all tools
        count = 0
        for tool_res in gen_results.values():
            if isinstance(tool_res, dict):
                if "molecules" in tool_res:
                    count += len(tool_res["molecules"])
                elif "count" in tool_res:
                    count += tool_res["count"]
                elif "smiles" in tool_res:
                    count += len(tool_res["smiles"])
                    
        print(f"[GeneratorAgent] Generation Complete. Generated {count} molecules.")
        
        # Append to agent_logs
        current_params = state.get("task_params", {}).copy()
        current_logs = current_params.get("agent_logs", "")
        redacted_results = {}
        if isinstance(gen_results, dict):
            for tool_name, tool_res in gen_results.items():
                if isinstance(tool_res, dict):
                    redacted_results[tool_name] = {k: v for k, v in tool_res.items() if k not in {"molecules", "smiles"}}
                else:
                    redacted_results[tool_name] = tool_res
        else:
            redacted_results = gen_results
        new_log = f"\n\n- **Generator Agent** (Loop {current_loop}):\n  - Model: {state.get('task_params', {}).get('tools', ['DiffSBDD'])[0]}\n  - Generated Count: {count}\n  - Results (Redacted): {json.dumps(redacted_results, indent=2)}"
        current_params["agent_logs"] = current_logs + new_log
        
        result["task_params"] = current_params
        
    # Increment plan step
    result["plan_step"] = state.get("plan_step", 0) + 1
        
    return result

def evaluator_node_wrapper(state: AgentState) -> Dict[str, Any]:
    print(f"\n[EvaluatorAgent] Evaluating Molecules...")
    result = evaluator_agent_node(state)
    
    if result.get("error"):
        print(f"[EvaluatorAgent] Error: {result.get('error')}")
    else:
        eval_results = result.get("results", {}).get("evaluation", {})
        qualified = eval_results.get("qualified_count", 0)
        print(f"[EvaluatorAgent] Evaluation Complete. Qualified: {qualified}")
        
        # Append to agent_logs
        current_params = state.get("task_params", {}).copy()
        current_logs = current_params.get("agent_logs", "")
        redacted_eval = {}
        if isinstance(eval_results, dict):
            redacted_eval = {k: v for k, v in eval_results.items() if k not in {"final_molecules", "top_molecules"}}
        else:
            redacted_eval = eval_results
        new_log = f"\n\n- **Evaluator Agent**:\n  - Qualified Candidates: {qualified}\n  - Feedback: {json.dumps(eval_results.get('feedback', {}), indent=2)}\n  - Results (Redacted): {json.dumps(redacted_eval, indent=2)}"
        current_params["agent_logs"] = current_logs + new_log
        
        result["task_params"] = current_params
        
    # --- Dynamic Routing Update (Loop Back Logic) ---
    # Moved from synthesis_node_wrapper to here because we want to loop back 
    # immediately after evaluation, regardless of whether synthesis is planned.
    next_agent = eval_results.get("next_agent")
    
    current_step = state.get("plan_step", 0)
    plan = state.get("plan", [])
    
    # Default: Advance
    next_step = current_step + 1
    
    if next_agent == "GeneratorAgent":
        loop_count = state.get("loop_count", 0)
        max_loops = int(os.getenv("MAX_GENERATION_LOOPS", 3))
        
        if loop_count < max_loops:
            # Check for feedback/missing count
            feedback = eval_results.get("feedback", {})
            missing = feedback.get("missing_count", 0)
            if missing > 0:
                # Update num_samples for next loop
                current_params = result.get("task_params", state.get("task_params", {}))
                current_params["num_samples"] = missing
                result["task_params"] = current_params
            
            print(f"[EvaluatorAgent] Looping back to Generator (Loop {loop_count}/{max_loops})...")
            try:
                gen_idx = plan.index("GeneratorAgent")
                next_step = gen_idx
            except ValueError:
                print("[EvaluatorAgent] GeneratorAgent not in plan! Cannot loop.")
        else:
            print("[EvaluatorAgent] Max loops reached. Continuing...")
            
    result["plan_step"] = next_step
        
    return result

def synthesis_node_wrapper(state: AgentState) -> Dict[str, Any]:
    print(f"\n[SynthesisAgent] Planning Synthesis Routes...")
    result = synthesis_agent_node(state)
    
    if result.get("error"):
        print(f"[SynthesisAgent] Error: {result.get('error')}")
    else:
        print(f"[SynthesisAgent] Synthesis Planning Complete.")
        synthesis_report = result.get("synthesis_report", "N/A")
        route_images = result.get("route_images", [])
        route_data_files = result.get("route_data_files", [])
        
        # Append to agent_logs
        current_params = state.get("task_params", {}).copy()
        current_logs = current_params.get("agent_logs", "")
        new_log = f"\n\n- **Synthesis Agent**:\n  - Report: {synthesis_report}\n  - Route Images: {len(route_images)}\n  - Route Data Files: {len(route_data_files)}"
        current_params["agent_logs"] = current_logs + new_log
        
        result["task_params"] = current_params

        merged_results = result.get("results", state.get("results", {}))
        if not isinstance(merged_results, dict):
            merged_results = {}
        merged_results["synthesis"] = {
            "synthesis_report": synthesis_report,
            "route_images": route_images,
            "route_data_files": route_data_files,
        }
        result["results"] = merged_results

    result["plan_step"] = state.get("plan_step", 0) + 1
        
    return result

def report_node_wrapper(state: AgentState) -> Dict[str, Any]:
    print("\n[ReportAgent] Generating Report...")
    # Prepare params for ReportAgent
    run_id = state.get("run_id")
    if run_id:
        eval_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "evaluation_results", run_id)
        
        # PREFER FILTERED CSV IF AVAILABLE
        # The user requested that the report (Safety Assessment, Full Dataset, Attachments) 
        # should be based on the filtered/qualified molecules, not the raw generation set.
        filtered_csv = os.path.join(eval_dir, "filtered_molecules.csv")
        final_csv = os.path.join(eval_dir, "final_evaluation.csv")
        
        if os.path.exists(filtered_csv):
            state["task_params"]["csv_path"] = filtered_csv
            print(f"[ReportAgent] Found Filtered CSV: {filtered_csv} (Using this for report)")
        elif os.path.exists(final_csv):
            state["task_params"]["csv_path"] = final_csv
            print(f"[ReportAgent] Filtered CSV not found. Using Full CSV: {final_csv}")
        else:
            print(f"[ReportAgent] Warning: No CSV found at {eval_dir}")
            
    # Note: agent_logs are now accumulated in task_params by previous agents
    # We don't need to reconstruct them here anymore, unless they are missing
    if "agent_logs" not in state["task_params"]:
        state["task_params"]["agent_logs"] = "No logs available."
        
    result = report_agent_node(state)
    print(f"[ReportAgent] Report Generated.")
    
    # Increment plan step
    result["plan_step"] = state.get("plan_step", 0) + 1
    
    # Mark task as complete to trigger frontend report display
    result["is_complete"] = True
    
    return result

# --- Graph Construction ---

workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("intent_agent", intent_node_wrapper)
workflow.add_node("coordinator_agent", coordinator_agent_node)
workflow.add_node("target_agent", target_node_wrapper)
workflow.add_node("generator_agent", generator_node_wrapper)
workflow.add_node("evaluator_agent", evaluator_node_wrapper)
workflow.add_node("synthesis_agent", synthesis_node_wrapper)
workflow.add_node("report_agent", report_node_wrapper)

# Add Edges
workflow.set_entry_point("intent_agent")

# Intent -> Coordinator (or End)
workflow.add_conditional_edges(
    "intent_agent",
    route_after_intent,
    {
        "coordinator_agent": "coordinator_agent",
        END: END
    }
)

# Coordinator -> Router -> First Agent
workflow.add_conditional_edges(
    "coordinator_agent",
    plan_router,
    {
        "target_agent": "target_agent",
        "generator_agent": "generator_agent",
        "evaluator_agent": "evaluator_agent",
        "synthesis_agent": "synthesis_agent",
        "report_agent": "report_agent",
        END: END
    }
)

# Agents -> Router -> Next Agent
agent_nodes = ["target_agent", "generator_agent", "evaluator_agent", "synthesis_agent", "report_agent"]

for node in agent_nodes:
    workflow.add_conditional_edges(
        node,
        plan_router,
        {
            "target_agent": "target_agent",
            "generator_agent": "generator_agent",
            "evaluator_agent": "evaluator_agent",
            "synthesis_agent": "synthesis_agent",
            "report_agent": "report_agent",
            END: END
        }
    )

# Compile
app = workflow.compile()

# --- Logging Setup ---

class DualLogger:
    """Helper to write to both stdout/stderr and a log file."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # Ensure real-time logging
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def setup_logging():
    """Redirects stdout and stderr to a timestamped log file."""
    # Create logs directory if not exists
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"workflow_run_{timestamp}.log")
    
    # Redirect stdout and stderr
    # Store original stdout in case we need to restore (not implemented here)
    sys.stdout = DualLogger(log_file)
    sys.stderr = sys.stdout # Redirect stderr to same log for unified view
    
    print(f"[System] Logging started. Output saved to: {log_file}")
    return log_file

# --- Main Interactive Loop ---

def interactive_workflow():
    print("\n=== Drug Discovery Agent Workflow (LangGraph Automatic Scheduling) ===")
    print("Type 'exit' or 'quit' to stop.\n")
    
    # Initialize RAG Agent
    print("[System] Initializing RAG Agent...")
    rag_agent = RAGAgent()
    
    history = []
    
    while True:
        try:
            user_input = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
            
        if user_input.lower() in ["exit", "quit"]:
            break
            
        if not user_input:
            continue
            
        run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        print(f"\n[System] Starting new run with ID: {run_id}")
        
        initial_state = AgentState(
            messages=history,
            user_input=user_input,
            intent=None,
            task_params={},
            current_agent="IntentAgent",
            results={},
            error=None,
            is_complete=False,
            run_id=run_id,
            loop_count=0
        )
        
        # Run the graph
        # We use invoke to run the full graph until END
        final_state = app.invoke(initial_state)
        
        # Update history from final state
        if final_state.get("messages"):
            history = final_state["messages"]
            
        # --- Store Experience in RAG ---
        # We only store if the task was substantial (e.g. generation/report) and successful
        intent = final_state.get("intent")
        if intent in ["generation", "report"] and not final_state.get("error"):
            try:
                print("\n[System] Archiving task experience to RAG Knowledge Base...")
                task_desc = final_state.get("user_input", "")
                
                # Get logs
                logs = final_state.get("task_params", {}).get("agent_logs", "")
                
                # Get summary
                summary = "Task completed successfully."
                # Try to get the last message content
                if final_state.get("messages"):
                    last_msg = final_state["messages"][-1]
                    if hasattr(last_msg, "content"):
                        summary = last_msg.content
                
                rag_agent.add_experience(task_desc, summary, logs)
            except Exception as e:
                print(f"[System] Warning: Failed to archive experience: {e}")

        print("\n(Workflow finished for this task.)")

if __name__ == "__main__":
    setup_logging()
    interactive_workflow()
