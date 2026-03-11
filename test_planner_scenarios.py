
import sys
import os
import json

# Add project root to sys.path
sys.path.append(os.getcwd())

from agent.planner import GraphPlanner

def test_plans():
    planner = GraphPlanner()
    
    scenarios = [
        ("generation", {"Target_ID"}, "Target-based Generation"),
        ("target-based", {"Target_ID"}, "Explicit Target-based"),
        ("de_novo", {"Target_ID"}, "De Novo Design"),
        ("optimization", {"Target_ID", "Molecules_SMILES"}, "Lead Optimization"),
        ("retrosynthesis", {"Molecules_SMILES"}, "Retrosynthesis"),
        ("evaluation", {"Molecules_SMILES"}, "Evaluation"),
        # Synthesis Variants
        ("generation", {"Target_ID"}, "Generation + Synthesis"), # Will pass run_retrosynthesis=True manually
        ("optimization", {"Target_ID", "Molecules_SMILES"}, "Optimization + Synthesis"),
    ]
    
    print(f"{'Intent':<20} | {'Inputs':<30} | {'Plan':<50}")
    print("-" * 110)
    
    for intent, inputs, desc in scenarios:
        task_params = {}
        if "Synthesis" in desc:
            task_params["run_retrosynthesis"] = True
            
        plan = planner.generate_plan(intent, inputs, task_params)
        print(f"{intent:<20} | {str(list(inputs)):<30} | {str(plan):<50}")

if __name__ == "__main__":
    test_plans()
