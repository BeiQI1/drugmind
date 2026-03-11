import json
import os

def create_fixed_kg():
    project_root = os.path.dirname(os.path.abspath(__file__))

    # 1. Define the correct Tool Registry (Code Implemented + Environment)
    tool_registry = {
        # --- TargetAgent Tools (Implemented) ---
        "ValidatePDB": {
            "agent": "TargetAgent",
            "category": "Target Preparation",
            "function": "Validates if a PDB ID exists and contains protein polymers.",
            "input": "pdb_id",
            "output": "bool",
            "usage": "Python"
        },
        "DownloadPDB": {
            "agent": "TargetAgent",
            "category": "Target Preparation",
            "function": "Downloads a PDB file from RCSB.",
            "input": "pdb_id",
            "output": "pdb_path",
            "usage": "Python"
        },
        "FetchMetadata": {
            "agent": "TargetAgent",
            "category": "Target Preparation",
            "function": "Fetches biological metadata (title, organism, function).",
            "input": "pdb_id",
            "output": "metadata_dict",
            "usage": "Python"
        },
        "PDBFixer": {
            "agent": "TargetAgent",
            "category": "Target Preparation",
            "function": "Repairs and cleans a PDB file (adds atoms/residues).",
            "input": "pdb_path",
            "output": "fixed_pdb_path",
            "usage": "Python (OpenMM)"
        },
        "CleanPDB": {
            "agent": "TargetAgent",
            "category": "Target Preparation",
            "function": "Simple cleaning of PDB file (removes water/ligands).",
            "input": "pdb_path",
            "output": "cleaned_pdb_path",
            "usage": "Python"
        },
        "ExtractLigand": {
            "agent": "TargetAgent",
            "category": "Target Preparation",
            "function": "Extracts the largest co-crystallized ligand.",
            "input": "pdb_path",
            "output": "ligand_sdf_path",
            "usage": "Python"
        },
        "fpocket": {
            "agent": "TargetAgent",
            "category": "Target Preparation",
            "function": "Detects binding pockets using fpocket.",
            "input": "pdb_path",
            "output": "pocket_center",
            "usage": "CLI"
        },
        "CalculateCenterFromResidues": {
            "agent": "TargetAgent",
            "category": "Target Preparation",
            "function": "Calculates center of mass for specific residues.",
            "input": "pdb_path, resi_list",
            "output": "center_coords",
            "usage": "Python"
        },
        # --- TargetAgent Tools (Environment / Supplementary) ---
        "AnalyzeStructureBioPython": {
            "agent": "TargetAgent",
            "category": "Target Preparation",
            "function": "Analyzes protein structure/sequence using BioPython.",
            "input": "pdb_path",
            "output": "analysis_dict",
            "usage": "Python (BioPython)"
        },
        "RunPdb4amber": {
            "agent": "TargetAgent",
            "category": "Target Preparation",
            "function": "Prepares PDB for Amber MD (pdb4amber).",
            "input": "pdb_path",
            "output": "amber_pdb",
            "usage": "CLI (AmberTools)"
        },
        "RunObabel": {
            "agent": "TargetAgent",
            "category": "Data Conversion",
            "function": "Converts chemical file formats (OpenBabel).",
            "input": "input_file, output_format",
            "output": "output_file",
            "usage": "CLI (OpenBabel)"
        },

        # --- GeneratorAgent Tools (Implemented) ---
        "RunDiffSBDD": {
            "agent": "GeneratorAgent",
            "category": "Generative Models",
            "function": "Generates molecules using DiffSBDD (Structure-based).",
            "input": "pdb_path, num_samples",
            "output": "sdf_path",
            "usage": "Service"
        },
        "RunDecompDiff": {
            "agent": "GeneratorAgent",
            "category": "Generative Models",
            "function": "Generates molecules using DecompDiff (Fragment-based).",
            "input": "pdb_path, num_samples",
            "output": "sdf_path",
            "usage": "Service"
        },
        "RunMiDi": {
            "agent": "GeneratorAgent",
            "category": "Generative Models",
            "function": "Generates molecules using MiDi.",
            "input": "pdb_path, smiles",
            "output": "sdf_path",
            "usage": "Service"
        },
        "RunGenMol": {
            "agent": "GeneratorAgent",
            "category": "Generative Models",
            "function": "Generates molecules using GenMol (Language Model).",
            "input": "prompt/smiles",
            "output": "smiles_list",
            "usage": "Service"
        },
        "RunDiffGui": {
            "agent": "GeneratorAgent",
            "category": "Generative Models",
            "function": "Generates molecules using DiffGui (Guided).",
            "input": "pdb_path",
            "output": "sdf_path",
            "usage": "Service"
        },

        # --- EvaluatorAgent Tools (Implemented) ---
        "CalculateQED": {
            "agent": "EvaluatorAgent",
            "category": "Scoring",
            "function": "Calculates QED score.",
            "input": "molecules",
            "output": "dataframe",
            "usage": "Python (RDKit)"
        },
        "CalculateSA": {
            "agent": "EvaluatorAgent",
            "category": "Scoring",
            "function": "Calculates Synthetic Accessibility score.",
            "input": "molecules",
            "output": "dataframe",
            "usage": "Python (RDKit)"
        },
        "CalculateLipinski": {
            "agent": "EvaluatorAgent",
            "category": "Scoring",
            "function": "Calculates Lipinski Rule of 5 properties.",
            "input": "molecules",
            "output": "dataframe",
            "usage": "Python (RDKit)"
        },
        "CalculateToxicity": {
            "agent": "EvaluatorAgent",
            "category": "Scoring",
            "function": "Calculates Toxicity/ADMET properties.",
            "input": "molecules",
            "output": "dataframe",
            "usage": "Python (TDC/RDKit)"
        },
        "RunDocking": {
            "agent": "EvaluatorAgent",
            "category": "Docking",
            "function": "Runs molecular docking (Vina/Gnina).",
            "input": "molecules, target_pdb, pocket_center",
            "output": "docking_scores",
            "usage": "CLI (Vina/Gnina)"
        },
        "ValidatePose": {
            "agent": "EvaluatorAgent",
            "category": "Analysis",
            "function": "Validates docked poses using PoseBusters.",
            "input": "molecules, target_pdb",
            "output": "validation_results",
            "usage": "Python"
        },
        "AnalyzeInteractions": {
            "agent": "EvaluatorAgent",
            "category": "Analysis",
            "function": "Analyzes protein-ligand interactions using PLIP.",
            "input": "molecules, target_pdb",
            "output": "interaction_counts",
            "usage": "Python (PLIP)"
        },
        # --- EvaluatorAgent Tools (Environment / Supplementary) ---
        "RunMeeko": {
            "agent": "EvaluatorAgent",
            "category": "Ligand Preparation",
            "function": "Prepares ligands for AutoDock Vina (SDF to PDBQT).",
            "input": "sdf_file",
            "output": "pdbqt_file",
            "usage": "Python (Meeko)"
        },
        "TrainQSAR": {
            "agent": "EvaluatorAgent",
            "category": "AI/ML",
            "function": "Trains a QSAR model using Scikit-Learn.",
            "input": "features, labels",
            "output": "model",
            "usage": "Python (Scikit-Learn)"
        },

        # --- SynthesisAgent Tools (Implemented) ---
        "RunRetrosynthesis": {
            "agent": "SynthesisAgent",
            "category": "Synthesis",
            "function": "Performs retrosynthesis planning (Syntheseus).",
            "input": "smiles",
            "output": "routes",
            "usage": "Python"
        },
        "RunAiZynth": {
            "agent": "SynthesisAgent",
            "category": "Synthesis",
            "function": "Runs AiZynthFinder retrosynthesis.",
            "input": "smiles",
            "output": "routes",
            "usage": "CLI (Conda)"
        },
        "GenerateSynthesisReport": {
            "agent": "SynthesisAgent",
            "category": "Synthesis",
            "function": "Generates HTML synthesis report.",
            "input": "results",
            "output": "html_report",
            "usage": "Python"
        },

        # --- ReportAgent Tools (Implemented) ---
        "GenerateDistributionPlot": {
            "agent": "ReportAgent",
            "category": "Visualization",
            "function": "Generates histogram/KDE plot.",
            "input": "data, column",
            "output": "image",
            "usage": "Python (Seaborn)"
        },
        "GenerateMoleculeGrid": {
            "agent": "ReportAgent",
            "category": "Visualization",
            "function": "Generates grid image of molecules.",
            "input": "molecules",
            "output": "image",
            "usage": "Python (RDKit)"
        },
        "WriteSection": {
            "agent": "ReportAgent",
            "category": "Reporting",
            "function": "Adds a section to the report.",
            "input": "content",
            "output": "none",
            "usage": "Python"
        },
        # --- ReportAgent Tools (Environment / Supplementary) ---
        "GeneratePDF": {
            "agent": "ReportAgent",
            "category": "Reporting",
            "function": "Converts HTML/Text to PDF using WeasyPrint/Xhtml2Pdf.",
            "input": "html_content",
            "output": "pdf_file",
            "usage": "Python (WeasyPrint)"
        },
        "AnalyzeGraph": {
            "agent": "ReportAgent",
            "category": "Analysis",
            "function": "Analyzes network properties using NetworkX.",
            "input": "graph_data",
            "output": "metrics",
            "usage": "Python (NetworkX)"
        }
    }

    # 2. Define Nodes
    nodes = {
        "IntentAgent": {
            "layer": "Interaction",
            "description": "Parses user input and manages dialogue state.",
            "tools": ["LangChain"],
            "inputs": ["User Text"],
            "outputs": ["Task Parameters"],
            "connections": {
                "to": ["TargetAgent", "RAGAgent"],
                "type": "Control Flow"
            }
        },
        "TargetAgent": {
            "layer": "Data_Preparation",
            "description": "Prepares protein target structures.",
            "tools": [
                "ValidatePDB", "DownloadPDB", "FetchMetadata", "PDBFixer", 
                "CleanPDB", "ExtractLigand", "fpocket", "CalculateCenterFromResidues", 
                "AnalyzeStructureBioPython", "RunPdb4amber", "RunObabel", "Finish"
            ],
            "inputs": ["PDB ID", "PDB File"],
            "outputs": ["Cleaned PDB", "Pocket Coordinates"],
            "connections": {
                "from": ["IntentAgent"],
                "to": ["GeneratorAgent", "EvaluatorAgent"],
                "type": "Data Flow"
            }
        },
        "RAGAgent": {
            "layer": "Data_Preparation",
            "description": "Retrieves external knowledge and literature.",
            "tools": ["FAISS", "PubChemPy"], # Kept simple
            "inputs": ["Search Query"],
            "outputs": ["Context"],
            "connections": {
                "from": ["IntentAgent"],
                "to": ["ReportAgent"],
                "type": "Information Flow"
            }
        },
        "GeneratorAgent": {
            "layer": "Generation",
            "description": "Generates candidate molecules using various models.",
            "tools": [
                "RunDiffSBDD", "RunDecompDiff", "RunMiDi", "RunGenMol", 
                "RunDiffGui", "Finish"
            ],
            "inputs": ["Pocket Info"],
            "outputs": ["Molecules SMILES"],
            "connections": {
                "from": ["TargetAgent"],
                "to": ["EvaluatorAgent"],
                "type": "Data Flow"
            }
        },
        "EvaluatorAgent": {
            "layer": "Evaluation",
            "description": "Evaluates generated molecules.",
            "tools": [
                "CalculateQED", "CalculateSA", "CalculateLipinski", "CalculateToxicity", 
                "RunDocking", "ValidatePose", "AnalyzeInteractions", "FilterMolecules", 
                "RunMeeko", "TrainQSAR", "Finish"
            ],
            "inputs": ["Molecules"],
            "outputs": ["Evaluation Report"],
            "connections": {
                "from": ["GeneratorAgent"],
                "to": ["SynthesisAgent", "ReportAgent"],
                "type": "Data Flow"
            }
        },
        "SynthesisAgent": {
            "layer": "Synthesis",
            "description": "Plans synthesis routes.",
            "tools": [
                "RunRetrosynthesis", "RunAiZynth", "GenerateSynthesisReport", "Finish"
            ],
            "inputs": ["Evaluated Molecules"],
            "outputs": ["Synthesis Plan"],
            "connections": {
                "from": ["EvaluatorAgent"],
                "to": ["ReportAgent"],
                "type": "Data Flow"
            }
        },
        "ReportAgent": {
            "layer": "Interaction",
            "description": "Generates final reports.",
            "tools": [
                "GenerateDistributionPlot", "GenerateMoleculeGrid", "WriteSection", 
                "GeneratePDF", "AnalyzeGraph", "Finish"
            ],
            "inputs": ["All Results"],
            "outputs": ["Final Report"],
            "connections": {
                "from": ["EvaluatorAgent", "SynthesisAgent", "RAGAgent"],
                "to": ["User"],
                "type": "Output Flow"
            }
        },
        "CoordinatorAgent": {
            "layer": "Reasoning_Control",
            "description": "Orchestrates the workflow using LangGraph.",
            "tools": ["LangGraph"],
            "inputs": ["State"],
            "outputs": ["Next Step"],
            "connections": {
                "from": ["All Agents"],
                "to": ["All Agents"],
                "type": "Control"
            }
        }
    }

    # 3. Workflow Logic (Injected from Enhanced)
    # NOTE: I am manually transcribing the Enhanced logic but fixing keys to match new node names.
    workflow_logic = {
        "data_dependencies": {
            "Target_ID": [],
            "Protein_Structure": ["TargetAgent"],
            "Pocket_Info": ["TargetAgent"],
            "Molecules_SMILES": ["GeneratorAgent"],
            "Evaluation_Report": ["EvaluatorAgent"],
            "Synthesis_Plan": ["SynthesisAgent"],
            "Final_Report": ["ReportAgent"]
        },
        "agent_requirements": {
            "TargetAgent": ["Target_ID"],
            "GeneratorAgent": ["Pocket_Info"],
            "EvaluatorAgent": ["Molecules_SMILES"],
            "SynthesisAgent": ["Molecules_SMILES"],
            "ReportAgent": ["Evaluation_Report"]
        },
        "conditional_requirements": {
            "EvaluatorAgent": {
                "optimization": ["Optimized_SMILES"]
            },
            "ReportAgent": {
                "default": ["Evaluation_Report"],
                "generation_with_synthesis": ["Evaluation_Report", "Synthesis_Plan"]
            },
            "GeneratorAgent": {
                "default": ["Pocket_Info"],
                "de_novo": []
            }
        },
        "forbidden_flows": [
            {
                "from": "TargetAgent",
                "to": "SynthesisAgent",
                "reason": "Cannot skip Generation."
            },
            {
                "from": "TargetAgent",
                "to": "EvaluatorAgent",
                "reason": "Cannot skip Generation."
            }
        ]
    }
    
    # 4. Assemble Final KG
    final_kg = {
        "meta": {
            "description": "Drug Design Multi-Agent System Knowledge Graph (Fixed & Supplemented)",
            "version": "3.0",
            "layers": ["Interaction", "Data_Preparation", "Generation", "Evaluation", "Synthesis", "Reasoning_Control"]
        },
        "tool_registry": tool_registry,
        "graph": {
            "nodes": nodes
        },
        "workflow_logic": workflow_logic
    }

    # 5. Write to file
    output_path = os.path.join(project_root, "drugtoolkg", "agent_kg_final.json")
    with open(output_path, "w") as f:
        json.dump(final_kg, f, indent=2)
    
    print(f"Successfully created {output_path}")

if __name__ == "__main__":
    create_fixed_kg()
