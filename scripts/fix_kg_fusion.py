import json
import os
import shutil

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KG_DIR = os.path.join(BASE_DIR, "drugtoolkg")
BASE_KG_PATH = os.path.join(KG_DIR, "agent_kg.json")
ENHANCED_KG_PATH = os.path.join(KG_DIR, "agent_kg_enhanced.json")
NEW_KG_PATH = os.path.join(KG_DIR, "agent_kg_new.json")

# 1. Real Tools Definition (Flattened for Registry)
# We need to map these to the format in agent_kg["tool_registry"]
# Format: Key = ToolName, Value = {category, source, function, input, output, usage}
REAL_TOOLS_METADATA = {
    # TargetAgent
    "ValidatePDB": {"category": "Target Preparation", "function": "Validates if a PDB ID exists and contains protein polymers.", "usage": "TargetAgent.validate_pdb_id"},
    "DownloadPDB": {"category": "Target Preparation", "function": "Downloads a PDB file from RCSB.", "usage": "TargetAgent.download_pdb"},
    "FetchMetadata": {"category": "Target Preparation", "function": "Fetches biological metadata for a PDB ID.", "usage": "TargetAgent.fetch_target_metadata"},
    "PDBFixer": {"category": "Target Preparation", "function": "Repairs and cleans a PDB file.", "usage": "TargetAgent.run_pdbfixer"},
    "CleanPDB": {"category": "Target Preparation", "function": "Simple cleaning of PDB file.", "usage": "TargetAgent.clean_pdb"},
    "ExtractLigand": {"category": "Target Preparation", "function": "Extracts the largest co-crystallized ligand.", "usage": "TargetAgent.extract_ligand_from_pdb"},
    "RDKit": {"category": "Target Preparation", "function": "Calculates geometric center of a ligand.", "usage": "TargetAgent.calculate_center_from_sdf"},
    "fpocket": {"category": "Target Preparation", "function": "Detects binding pockets.", "usage": "TargetAgent.run_fpocket"},
    "CalculateCenterFromResidues": {"category": "Target Preparation", "function": "Calculates center of mass for specific residues.", "usage": "TargetAgent.calculate_center_from_residues"},
    
    # GeneratorAgent
    "RunDiffSBDD": {"category": "Generative Models", "function": "Structure-based molecule generation.", "usage": "GeneratorAgent.run_diffsbdd"},
    "RunDiffGui": {"category": "Generative Models", "function": "Property-guided molecule generation.", "usage": "GeneratorAgent.run_diffgui"},
    "RunGenMol": {"category": "Generative Models", "function": "SMILES/Language-based molecule generation.", "usage": "GeneratorAgent.run_genmol"},
    "RunMiDi": {"category": "Generative Models", "function": "Multi-modal molecule generation.", "usage": "GeneratorAgent.run_midi"},
    "RunDecompDiff": {"category": "Generative Models", "function": "Fragment-based molecule generation.", "usage": "GeneratorAgent.run_decompdiff"},
    
    # EvaluatorAgent
    "CalculateQED": {"category": "Evaluation", "function": "Calculates QED score.", "usage": "EvaluatorAgent.calculate_qed"},
    "CalculateSA": {"category": "Evaluation", "function": "Calculates SA score.", "usage": "EvaluatorAgent.calculate_sa"},
    "CalculateLipinski": {"category": "Evaluation", "function": "Calculates Lipinski rules.", "usage": "EvaluatorAgent.calculate_lipinski"},
    "CalculateToxicity": {"category": "Evaluation", "function": "Calculates Toxicity metrics.", "usage": "EvaluatorAgent.calculate_toxicity"},
    "CalculateDiversity": {"category": "Evaluation", "function": "Calculates diversity metrics.", "usage": "EvaluatorAgent.calculate_diversity"},
    "RunStandardEvaluation": {"category": "Evaluation", "function": "Runs standard battery of tests.", "usage": "EvaluatorAgent.run_standard_evaluation"},
    "RunDocking": {"category": "Evaluation", "function": "Runs molecular docking.", "usage": "EvaluatorAgent.run_docking"},
    "ValidatePose": {"category": "Evaluation", "function": "Validates docked poses.", "usage": "EvaluatorAgent.validate_pose"},
    "AnalyzeInteractions": {"category": "Evaluation", "function": "Analyzes protein-ligand interactions.", "usage": "EvaluatorAgent.analyze_interactions"},
    "FilterMolecules": {"category": "Evaluation", "function": "Filters molecules based on criteria.", "usage": "EvaluatorAgent.filter_molecules"},
    
    # SynthesisAgent
    "LoadCandidates": {"category": "Synthesis", "function": "Loads molecules for synthesis planning.", "usage": "SynthesisAgent.load_candidates"},
    "SelectTopN": {"category": "Synthesis", "function": "Selects top candidates.", "usage": "SynthesisAgent.select_top_n"},
    "RunRetrosynthesis": {"category": "Synthesis", "function": "Runs retrosynthesis (Syntheseus).", "usage": "SynthesisAgent.run_retrosynthesis"},
    "RunAiZynth": {"category": "Synthesis", "function": "Runs AiZynthFinder.", "usage": "SynthesisAgent.run_aizynth"},
    "GenerateSynthesisReport": {"category": "Synthesis", "function": "Generates synthesis report.", "usage": "SynthesisAgent.generate_synthesis_report"},
    "AnalyzeRouteComplexity": {"category": "Synthesis", "function": "Analyzes route complexity.", "usage": "SynthesisAgent.analyze_route_complexity"},
    
    # ReportAgent
    "LoadData": {"category": "Reporting", "function": "Loads data for reporting.", "usage": "ReportAgent.load_data"},
    "RetrieveContext": {"category": "Reporting", "function": "Retrieves background context.", "usage": "ReportAgent.retrieve_context"},
    "GenerateDistributionPlot": {"category": "Reporting", "function": "Generates distribution plots.", "usage": "ReportAgent.generate_distribution_plot"},
    "GenerateCorrelationPlot": {"category": "Reporting", "function": "Generates correlation plots.", "usage": "ReportAgent.generate_correlation_plot"},
    "GenerateHeatmap": {"category": "Reporting", "function": "Generates heatmaps.", "usage": "ReportAgent.generate_heatmap"},
    "GenerateMoleculeGrid": {"category": "Reporting", "function": "Generates molecule grid images.", "usage": "ReportAgent.generate_molecule_grid"},
    "GenerateSimilarityNetwork": {"category": "Reporting", "function": "Generates similarity networks.", "usage": "ReportAgent.generate_similarity_network"},
    "GetTable": {"category": "Reporting", "function": "Generates data tables.", "usage": "ReportAgent.get_table"},
    "AssessToxicity": {"category": "Reporting", "function": "Assesses toxicity risks.", "usage": "ReportAgent.assess_toxicity"},
    "WriteSection": {"category": "Reporting", "function": "Writes report sections.", "usage": "ReportAgent.write_section"},
    
    # Shared 'Finish' tool
    "Finish": {"category": "Control", "function": "Completes the agent's task.", "usage": "Agent.finish"}
}

AGENT_TOOLS_MAPPING = {
    "TargetAgent": ["ValidatePDB", "DownloadPDB", "FetchMetadata", "PDBFixer", "CleanPDB", "ExtractLigand", "RDKit", "fpocket", "CalculateCenterFromResidues", "Finish"],
    "GeneratorAgent": ["RunDiffSBDD", "RunDiffGui", "RunGenMol", "RunMiDi", "RunDecompDiff", "Finish"],
    "EvaluatorAgent": ["CalculateQED", "CalculateSA", "CalculateLipinski", "CalculateToxicity", "CalculateDiversity", "RunStandardEvaluation", "RunDocking", "ValidatePose", "AnalyzeInteractions", "FilterMolecules", "Finish"],
    "SynthesisAgent": ["LoadCandidates", "SelectTopN", "RunRetrosynthesis", "RunAiZynth", "GenerateSynthesisReport", "AnalyzeRouteComplexity", "Finish"],
    "ReportAgent": ["LoadData", "RetrieveContext", "GenerateDistributionPlot", "GenerateCorrelationPlot", "GenerateHeatmap", "GenerateMoleculeGrid", "GenerateSimilarityNetwork", "GetTable", "AssessToxicity", "WriteSection", "Finish"]
}

def main():
    print("Loading KGs...")
    with open(BASE_KG_PATH, 'r') as f:
        base_kg = json.load(f)
    
    with open(ENHANCED_KG_PATH, 'r') as f:
        enhanced_kg = json.load(f)
    
    new_kg = base_kg.copy()
    
    # 1. Update Tool Registry
    # We keep existing tools in registry but update/add ours
    registry = new_kg.get("tool_registry", {})
    for tool_name, metadata in REAL_TOOLS_METADATA.items():
        # Preserve existing metadata if we don't have it (like source/input/output), or overwrite?
        # Let's overwrite/merge
        if tool_name not in registry:
            registry[tool_name] = {}
        
        registry[tool_name].update({
            "category": metadata["category"],
            "function": metadata["function"],
            "usage": metadata["usage"],
            "source": "DrugToolAgent Codebase" # Mark as verified
        })
    new_kg["tool_registry"] = registry
    print(f"Updated Tool Registry with {len(REAL_TOOLS_METADATA)} real tools.")

    # 2. Update Agents (Nodes)
    nodes = new_kg.get("graph", {}).get("nodes", {})
    
    # Fix GeneratorAgents -> GeneratorAgent
    if "GeneratorAgents" in nodes:
        print("Renaming GeneratorAgents -> GeneratorAgent")
        nodes["GeneratorAgent"] = nodes.pop("GeneratorAgents")
    
    # Remove Invalid Agents
    invalid_agents = ["ConditionAgent", "TrapAgent", "TestAgent"]
    for inv in invalid_agents:
        if inv in nodes:
            print(f"Removing invalid agent: {inv}")
            del nodes[inv]

    # Fix connections in remaining nodes
    for agent_name, agent_data in nodes.items():
        if "connections" in agent_data and "to" in agent_data["connections"]:
            connections = agent_data["connections"]["to"]
            new_connections = []
            for conn in connections:
                if conn == "GeneratorAgents":
                    new_connections.append("GeneratorAgent")
                elif conn in invalid_agents:
                    continue
                else:
                    new_connections.append(conn)
            agent_data["connections"]["to"] = new_connections
            
    # Update Tools List for Each Agent
    for agent_name, tool_list in AGENT_TOOLS_MAPPING.items():
        if agent_name in nodes:
            nodes[agent_name]["tools"] = tool_list
            print(f"Updated tools for {agent_name}")
        else:
            # Create if missing (CoordinatorAgent might be there, we skip it here as it has no tools in mapping)
            print(f"Warning: {agent_name} not found in nodes, creating skeleton.")
            nodes[agent_name] = {
                "layer": "Operation",
                "description": f"Agent for {agent_name}",
                "tools": tool_list,
                "inputs": [],
                "outputs": [],
                "connections": {"to": [], "from": [], "type": "Data Flow"}
            }
            
    new_kg["graph"]["nodes"] = nodes
    
    # 3. Update Links
    # Fix GeneratorAgents references
    links = new_kg.get("links", [])
    new_links = []
    for link in links:
        s = link.get("source", "")
        t = link.get("target", "")
        
        # Rename
        s = s.replace("GeneratorAgents", "GeneratorAgent")
        t = t.replace("GeneratorAgents", "GeneratorAgent")
        
        # Filter invalid
        if "ConditionAgent" in s or "ConditionAgent" in t: continue
        if "TrapAgent" in s or "TrapAgent" in t: continue
        
        link["source"] = s
        link["target"] = t
        new_links.append(link)
    new_kg["links"] = new_links
    
    # 4. Inject Workflow Logic
    print("Injecting workflow logic...")
    workflow = enhanced_kg.get("workflow_logic", {})
    sanitized_workflow = {
        "data_dependencies": {}, 
        "agent_requirements": {},
        "conditional_requirements": {}
    }
    
    # Sanitize Dependencies
    if "data_dependencies" in workflow:
        for k, v in workflow["data_dependencies"].items():
            # Key might be "Molecules_SMILES" or Agent Name
            clean_v = []
            for item in v:
                if item == "GeneratorAgents": item = "GeneratorAgent"
                if item in ["ConditionAgent", "TrapAgent"]: continue
                clean_v.append(item)
            sanitized_workflow["data_dependencies"][k] = clean_v
            
    # Sanitize Requirements
    if "agent_requirements" in workflow:
        for k, v in workflow["agent_requirements"].items():
            key = k
            if key == "GeneratorAgents": key = "GeneratorAgent"
            if key in ["ConditionAgent", "TrapAgent"]: continue
            
            sanitized_workflow["agent_requirements"][key] = v

    # Sanitize Conditional Requirements
    if "conditional_requirements" in workflow:
        for k, v in workflow["conditional_requirements"].items():
            key = k
            if key == "GeneratorAgents": key = "GeneratorAgent"
            if key in ["ConditionAgent", "TrapAgent"]: continue
            
            sanitized_workflow["conditional_requirements"][key] = v
            
    new_kg["workflow_logic"] = sanitized_workflow
    
    # 5. Save
    print(f"Saving to {NEW_KG_PATH}...")
    with open(NEW_KG_PATH, 'w') as f:
        json.dump(new_kg, f, indent=2)
    print("Done.")

if __name__ == "__main__":
    main()
