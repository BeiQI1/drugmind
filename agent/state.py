from typing import TypedDict, Annotated, List, Dict, Any, Optional, Union
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    # Conversation history
    messages: Annotated[List[BaseMessage], add_messages]
    
    # The raw user input text
    user_input: str
    
    # Classified intent (e.g., "generate_molecule", "docking", "synthesis_planning")
    intent: Optional[str]
    
    # Extracted parameters for the task (e.g., {"pdb_id": "1abc", "smiles": "CCO"})
    task_params: Dict[str, Any]
    
    # The current active agent name
    current_agent: str
    
    # Shared blackboard for results from different agents
    results: Dict[str, Any]
    
    # Error tracking
    error: Optional[str]
    
    # Unique Run ID
    run_id: Optional[str]
    
    # Loop counter for generation/optimization cycles
    loop_count: int
    
    # Dynamic Execution Plan
    plan: Optional[List[str]]
    plan_step: int
    
    # Flag to indicate if the task is complete
    is_complete: bool
