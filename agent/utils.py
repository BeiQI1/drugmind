import json
import os
from typing import Dict, Any, List

class KnowledgeGraphLoader:
    _instance = None
    _kg_data = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(KnowledgeGraphLoader, cls).__new__(cls)
        return cls._instance

    def load_kg(self, kg_path: str = "drugtoolkg/agent_kg.json") -> Dict[str, Any]:
        """Loads the Knowledge Graph JSON."""
        if self._kg_data:
            return self._kg_data
            
        # Resolve absolute path if needed, assuming run from project root
        if not os.path.isabs(kg_path):
            # Try to find the file relative to the current working directory or this file
            base_path = os.getcwd()
            full_path = os.path.join(base_path, kg_path)
            if not os.path.exists(full_path):
                # Fallback: try to find it relative to this file's location
                # This file is in agent/utils.py, so drugtoolkg is ../drugtoolkg
                current_dir = os.path.dirname(os.path.abspath(__file__))
                full_path = os.path.join(current_dir, "../drugtoolkg/agent_kg.json")
        else:
            full_path = kg_path

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Knowledge Graph file not found at: {full_path}")

        with open(full_path, 'r', encoding='utf-8') as f:
            self._kg_data = json.load(f)
        
        return self._kg_data

    def get_agent_node(self, agent_name: str) -> Dict[str, Any]:
        """Retrieves the configuration/node for a specific agent."""
        kg = self.load_kg()
        nodes = kg.get("graph", {}).get("nodes", {})
        return nodes.get(agent_name, {})

    def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """Retrieves metadata for a specific tool."""
        kg = self.load_kg()
        registry = kg.get("tool_registry", {})
        return registry.get(tool_name, {})

    def query_agent_tools(self, agent_name: str) -> List[Dict[str, Any]]:
        """Returns a list of full tool definitions for a given agent."""
        agent_node = self.get_agent_node(agent_name)
        tool_names = agent_node.get("tools", [])
        tools_info = []
        for name in tool_names:
            info = self.get_tool_info(name)
            if info:
                info['name'] = name
                tools_info.append(info)
        return tools_info
