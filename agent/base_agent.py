import os
from typing import Any, Dict, Optional
from langchain_core.messages import SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from agent.utils import KnowledgeGraphLoader
from agent.state import AgentState
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

class BaseAgent:
    def __init__(self, agent_name: str, model: Optional[BaseChatModel] = None):
        self.agent_name = agent_name
        self.kg_loader = KnowledgeGraphLoader()
        self.node_info = self.kg_loader.get_agent_node(agent_name)
        
        # If model is not provided, initialize it using environment variables
        if model is None:
            self.model = self._init_llm()
        else:
            self.model = model

    def _init_llm(self) -> ChatOpenAI:
        """Initializes the LLM based on agent-specific or global environment variables."""
        # Convert agent name to env var prefix (e.g., IntentAgent -> INTENT_AGENT)
        prefix = self.agent_name.upper().replace(" ", "_")
        
        # 1. API Key
        api_key = os.getenv(f"{prefix}_LLM_API_KEY") or os.getenv("GLOBAL_LLM_API_KEY") or os.getenv("AGENT_LLM_API_KEY")
        
        # 2. Base URL
        base_url = os.getenv(f"{prefix}_LLM_API_BASE") or os.getenv("GLOBAL_LLM_API_BASE") or os.getenv("LLM_API_BASE")
        if base_url and not base_url.endswith("/v1") and "kfcv50" in base_url:
             base_url += "/v1"
             
        # 3. Model Name
        model_name = os.getenv(f"{prefix}_LLM_MODEL") or os.getenv("GLOBAL_LLM_MODEL") or "gpt-5"
        
        if not api_key:
            print(f"Warning: No API Key found for {self.agent_name} (checked {prefix}_LLM_API_KEY and GLOBAL_LLM_API_KEY)")
            # Fallback to empty string to avoid validation error immediately, 
            # but it will fail if used.
            api_key = "missing_api_key"

        # Check for reasoning models that enforce temperature=1
        # gpt-5.1-chat-latest, o1-preview, o1-mini etc. often require temperature=1
        temperature = 0
        if model_name in ["gpt-5.1-chat-latest", "o1-preview", "o1-mini"]:
            temperature = 1

        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_retries=5,
            timeout=60
        )

    def get_system_prompt(self) -> str:
        """Generates a standard system prompt based on KG definition."""
        description = self.node_info.get("description", "")
        tools = self.node_info.get("tools", [])
        inputs = self.node_info.get("inputs", [])
        outputs = self.node_info.get("outputs", [])
        
        prompt = f"You are the {self.agent_name}.\n"
        prompt += f"Role Description: {description}\n"
        prompt += f"You have access to the following tools: {', '.join(tools)}\n"
        prompt += f"You accept inputs: {', '.join(inputs)}\n"
        prompt += f"You produce outputs: {', '.join(outputs)}\n"
        
        # Add specific knowledge base info if available
        kb = self.node_info.get("knowledge_base", {})
        if kb:
            prompt += f"Knowledge Base Context: {kb}\n"

        prompt += "Language: English only. Always respond in English.\n"
            
        return prompt

    def search_tools_in_kg(self, query_type: str = None) -> list:
        """
        Allows any agent to search the KG for tools matching specific criteria.
        This replaces/augments basic RAG search with structured graph queries.
        """
        # This is a placeholder for the actual KG traversal logic.
        # In a real implementation, we would query self.kg_loader.graph
        # For now, we return the tools assigned to this agent in the KG.
        # But we filter them if query_type is provided.
        
        all_tools = self.kg_loader.query_agent_tools(self.agent_name)
        
        if query_type:
             # Simple filter simulation
             filtered = [t for t in all_tools if query_type.lower() in str(t).lower()]
             return filtered
             
        return all_tools


    def run(self, state: AgentState) -> Dict[str, Any]:
        """
        Main execution method to be implemented by subclasses or used directly.
        Returns a dictionary of updates to the state.
        """
        raise NotImplementedError("Subclasses must implement run()")
