import json

def merge_kg():
    # Load original agent_kg (Base)
    with open('drugtoolkg/agent_kg.json', 'r') as f:
        agent_kg = json.load(f)

    # Load enhanced kg (Logic Source)
    with open('drugtoolkg/agent_kg_enhanced.json', 'r') as f:
        enhanced_kg = json.load(f)

    # 1. Update Meta
    agent_kg['meta']['version'] = '3.0-Merged'
    agent_kg['meta']['description'] = 'Merged KG: agent_kg tools + enhanced orchestration logic'

    # 2. Inject Workflow Logic
    agent_kg['workflow_logic'] = enhanced_kg['workflow_logic']

    # 3. Process Nodes
    original_nodes = agent_kg['graph']['nodes']
    enhanced_nodes = enhanced_kg['tool_registry'] # In enhanced, nodes are under tool_registry

    new_nodes = {}

    # Define the mapping/renaming and retention policy
    # Keys: Original Name in agent_kg
    # Values: New Name (or None to delete)
    node_mapping = {
        "IntentAgent": "IntentAgent",
        "TargetAgent": "TargetAgent",
        "RAGAgent": "RAGAgent",
        "ConditionAgent": None, # Delete
        "GeneratorAgents": "GeneratorAgent", # Rename
        "EvaluatorAgent": "EvaluatorAgent",
        "CoordinatorAgent": "CoordinatorAgent",
        "SynthesisAgent": "SynthesisAgent",
        "ReportAgent": "ReportAgent",
        "TestAgent": None # Delete
    }

    # Helper to get enhanced node data if available
    def get_enhanced_data(agent_name):
        # Enhanced uses "GeneratorAgent" keys directly
        return enhanced_nodes.get(agent_name, {})

    for old_name, new_name in node_mapping.items():
        if new_name is None:
            continue
        
        # Start with original node data (to keep description, knowledge_base, tools)
        node_data = original_nodes.get(old_name, {}).copy()
        
        # If it's a rename (GeneratorAgents -> GeneratorAgent), ensure we don't lose data
        if not node_data and old_name == "GeneratorAgents":
             # This shouldn't happen based on the file, but good for safety
             pass

        # Fetch enhanced data using the NEW name (since enhanced uses 'GeneratorAgent')
        enhanced_data = get_enhanced_data(new_name)
        
        # Inject SOP Schema from Enhanced
        if 'sop_schema' in enhanced_data:
            node_data['sop_schema'] = enhanced_data['sop_schema']

        # Update Connections based on Enhanced Logic (Manual fix for consistency)
        # We derive this from the workflow logic roughly
        current_connections = node_data.get('connections', {})
        current_to = current_connections.get('to', [])
        
        new_to = []
        if new_name == "IntentAgent":
             new_to = ["TargetAgent", "RAGAgent"] # Removed ConditionAgent
        elif new_name == "TargetAgent":
             new_to = ["GeneratorAgent", "EvaluatorAgent"] # Removed ConditionAgent, Renamed Generator
        elif new_name == "GeneratorAgent":
             new_to = ["EvaluatorAgent", "CoordinatorAgent"]
        elif new_name == "EvaluatorAgent":
             new_to = ["CoordinatorAgent", "ReportAgent"] # Removed SynthesisAgent? Wait, enhanced allows synthesis
             # Check dependencies: Synthesis needs Molecules. Generator provides them.
             # Evaluator provides Evaluation_Report. Synthesis doesn't explicitly depend on Evaluator in 'agent_requirements' (depends on Molecules).
             # But flow usually goes Gen -> Eval -> Coord -> (Stop) -> Synthesis -> Report
             pass 
        elif new_name == "CoordinatorAgent":
             new_to = ["GeneratorAgent"] # Removed ConditionAgent
        elif new_name == "SynthesisAgent":
             new_to = ["ReportAgent"]
        elif new_name == "RAGAgent":
             new_to = ["ReportAgent"] # Removed ConditionAgent
        
        # Only update if we manually defined it, otherwise keep (but filter deleted agents)
        if new_to:
            node_data['connections'] = node_data.get('connections', {})
            node_data['connections']['to'] = new_to
        else:
            # Filter out deleted agents from existing connections
            valid_targets = ["IntentAgent", "TargetAgent", "RAGAgent", "GeneratorAgent", 
                             "EvaluatorAgent", "CoordinatorAgent", "SynthesisAgent", "ReportAgent"]
            
            # Handle the plural->singular rename in connections
            cleaned_to = []
            for t in node_data.get('connections', {}).get('to', []):
                if t == "GeneratorAgents": t = "GeneratorAgent"
                if t in valid_targets:
                    cleaned_to.append(t)
            node_data['connections']['to'] = cleaned_to

        new_nodes[new_name] = node_data

    agent_kg['graph']['nodes'] = new_nodes

    # 4. Save
    with open('drugtoolkg/agent_kg.json', 'w') as f:
        json.dump(agent_kg, f, indent=2)

if __name__ == "__main__":
    merge_kg()
