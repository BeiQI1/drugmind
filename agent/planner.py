import json
import os
import networkx as nx
from typing import List, Dict, Set

class GraphPlanner:
    def __init__(self, kg_path: str = None):
        if kg_path is None:
            kg_path = os.path.join(os.path.dirname(__file__), "../drugtoolkg/agent_kg.json")
        
        with open(kg_path, 'r') as f:
            self.kg = json.load(f)
            
        self.dependencies = self.kg["workflow_logic"]["data_dependencies"]
        self.requirements = self.kg["workflow_logic"]["agent_requirements"]
        
    def generate_plan(self, intent: str, current_state_keys: Set[str], task_params: Dict = None) -> List[str]:
        """
        Generates a plan (list of agents) using backward chaining based on KG dependencies.
        Enhanced to support conditional logic and dynamic resource mapping.
        """
        if task_params is None:
            task_params = {}
            
        target_resource = self._map_intent_to_resource(intent, task_params)
        if not target_resource:
            return [] # Unknown intent
            
        plan = []
        visited_resources = set()
        
        # Backward Chaining
        needed_agents = []
        
        def resolve_dependency(resource):
            if resource in visited_resources:
                return
            visited_resources.add(resource)
            
            producers = self.dependencies.get(resource, [])
            
            if not producers:
                return
            
            # If resource is in current_state, we usually skip production.
            # But for the Target Resource, we always want to produce it (unless it's just a query?)
            # Let's assume we re-run if it's the goal.
            if resource != target_resource and resource in current_state_keys:
                return
                
            agent_name = producers[0] 
            
            # --- Dynamic Requirements Logic ---
            # Check for conditional requirements based on intent
            reqs = []
            
            # 1. Try to find specific requirements for this intent in "conditional_requirements"
            cond_reqs = self.kg.get("workflow_logic", {}).get("conditional_requirements", {})
            agent_conds = cond_reqs.get(agent_name, {})
            
            # Normalize intent for matching
            intent_norm = intent.lower().replace(" ", "_").replace("-", "_")
            
            # Map aliases & Handle Modes
            if intent_norm == "generation":
                mode = task_params.get("mode", "structure_based")
                if "de_novo" in mode or "denovo" in mode:
                    intent_norm = "de_novo"
                else:
                    intent_norm = "structure_based"
                
            # Handle "With Synthesis" variants for ReportAgent
            # The intent might be "generation", but if we are resolving "Final_Report_With_Synthesis",
            # we need the "generation_with_synthesis" requirements for ReportAgent.
            # But "intent" variable is just "generation".
            # We can detect this by checking the target_resource or checking the flag again.
            
            # Check for synthesis flag
            include_synthesis = task_params.get("run_retrosynthesis", False) or \
                                task_params.get("include_synthesis", False) or \
                                "synthesis" in intent_norm
                                
            # If we are resolving ReportAgent and synthesis is requested, look for the combined key
            lookup_key = intent_norm
            if include_synthesis and agent_name == "ReportAgent":
                 # Construct key like "generation_with_synthesis"
                 lookup_key = f"{intent_norm}_with_synthesis"
                 
            # Retrosynthesis specific mapping
            if intent_norm in ["retrosynthesis", "retrosynthesis_planning", "synthesis_planning"]:
                 lookup_key = "retrosynthesis"

            if lookup_key in agent_conds:
                reqs = agent_conds[lookup_key]
            elif "default" in agent_conds:
                reqs = agent_conds["default"]
            else:
                # Fallback to standard requirements
                reqs = self.requirements.get(agent_name, [])
            
            # Recursively resolve requirements
            for req in reqs:
                resolve_dependency(req)
            
            # Add agent to plan (Topological Sort: Dependencies first)
            if agent_name not in needed_agents:
                needed_agents.append(agent_name)
                
        # Handle special composite targets (like Final_Report_With_Synthesis)
        # We need to manually resolve its dependencies because it's a "virtual" resource 
        # that maps to ReportAgent but requires multiple inputs.
        
        # Actually, let's make it simpler:
        # If target is "Final_Report_With_Synthesis", we know it needs Eval + Synth.
        # But our resolve_dependency logic relies on "producers".
        # If "Final_Report_With_Synthesis" is produced by "ReportAgent", 
        # then "ReportAgent" needs to know it requires [Eval, Synth] for THIS target.
        # So we can use the same conditional logic if we map the intent correctly.
        
        resolve_dependency(target_resource)
        
        return needed_agents

    def _map_intent_to_resource(self, intent: str, task_params: Dict = None) -> str:
        if task_params is None:
            task_params = {}
        # Normalize intent
        intent_norm = intent.lower().replace(" ", "_").replace("-", "_")
        
        # Check for synthesis flag
        include_synthesis = task_params.get("run_retrosynthesis", False) or \
                            task_params.get("include_synthesis", False) or \
                            "synthesis" in intent_norm
                            
        # 1. Retrosynthesis Only
        if intent_norm in ["retrosynthesis", "retrosynthesis_planning", "synthesis_planning"]:
            return "Synthesis_Report"
            
        # 2. Evaluation Only
        if intent_norm in ["evaluation", "property_evaluation"]:
            if include_synthesis:
                return "Final_Report_With_Synthesis"
            return "Final_Report"

        # 3. Generation / Optimization / De Novo
        if intent_norm in ["generation", "structure_based", "structure_based_generation", 
                          "target_based", "target_based_generation",
                          "de_novo", "de_novo_generation", "de_novo_design",
                          "optimization", "molecule_optimization", "lead_optimization"]:
            
            if include_synthesis:
                return "Final_Report_With_Synthesis"
            return "Final_Report"
            
        return None
