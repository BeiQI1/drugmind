
import os
import sys
import json
import traceback

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.planner import GraphPlanner

def run_tests():
    planner = GraphPlanner()
    print("GraphPlanner initialized with agent_kg.json")
    
    tests = [
        {
            "name": "Standard Generation (Default)",
            "intent": "generation",
            "params": {},
            "expected_agents": ["TargetAgent", "GeneratorAgent", "EvaluatorAgent", "ReportAgent"]
        },
        {
            "name": "Generation with Synthesis",
            "intent": "generation",
            "params": {"run_retrosynthesis": True},
            "expected_agents": ["TargetAgent", "GeneratorAgent", "EvaluatorAgent", "SynthesisAgent", "ReportAgent"]
        },
        {
            "name": "Optimization (Evaluator-driven)",
            "intent": "optimization",
            "params": {},
            "expected_agents": ["GeneratorAgent", "EvaluatorAgent", "ReportAgent"]
        },
        {
            "name": "De Novo Generation (Explicit Intent)",
            "intent": "de_novo",
            "params": {},
            "expected_agents": ["GeneratorAgent", "EvaluatorAgent", "ReportAgent"]
        },
        {
            "name": "De Novo Generation (via Mode Param)",
            "intent": "generation",
            "params": {"mode": "de_novo"},
            "expected_agents": ["GeneratorAgent", "EvaluatorAgent", "ReportAgent"]
        },
        {
            "name": "Retrosynthesis Only",
            "intent": "retrosynthesis",
            "params": {"smiles": "CCO"},
            "expected_agents": ["SynthesisAgent", "ReportAgent"]
        },
        {
            "name": "Evaluation Only (With SMILES)",
            "intent": "evaluation",
            "params": {"smiles": "CCO"},
            "expected_agents": ["EvaluatorAgent", "ReportAgent"]
        }
    ]
    
    passed_count = 0
    
    for i, test in enumerate(tests):
        print(f"\n--- Test {i+1}: {test['name']} ---")
        try:
            current_state = test.get("current_state", set())
            
            # Mimic CoordinatorAgent logic
            params = test.get("params", {})
            if "pdb_id" in params or "target_name" in params:
                current_state.add("Target_ID")
            if "smiles" in params or "molecules" in params:
                current_state.add("Molecules_SMILES")
                
            plan = planner.generate_plan(test["intent"], current_state_keys=current_state, task_params=test["params"])
            print(f"Plan: {plan}")
            
            expected = test["expected_agents"]
            if plan == expected:
                print("✅ PASSED")
                passed_count += 1
            else:
                print(f"❌ FAILED. Expected {expected}")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
            traceback.print_exc()

    print(f"\nTotal Tests: {len(tests)}")
    print(f"Passed: {passed_count}")

if __name__ == "__main__":
    run_tests()
