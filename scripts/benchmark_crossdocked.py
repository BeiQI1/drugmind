import os
import sys
import torch
import pandas as pd
import json
import time
from tqdm import tqdm
from langchain_core.messages import HumanMessage

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.state import AgentState
# Import the REAL app from the interactive workflow
from agent.interactive_workflow import app

import argparse

def run_benchmark():
    parser = argparse.ArgumentParser(description="Run CrossDocked Benchmark")
    parser.add_argument("--output_dir", type=str, help="Directory to save results. If it exists, will attempt to resume.")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of molecules to generate per target (default: 20)")
    args = parser.parse_args()

    print("=== Starting CrossDocked Benchmark (Authentic Workflow) ===")
    
    # 1. Load Test Data
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test", "dataset", "test_data.pt")
    if not os.path.exists(data_path):
        print(f"Error: Test data not found at {data_path}")
        return
        
    print(f"Loading test data from {data_path}...")
    try:
        test_data = torch.load(data_path)
    except Exception as e:
        print(f"Error loading test data: {e}")
        return

    # Check data format
    if isinstance(test_data, dict):
        # Convert dict to list of (pocket, ligand) tuples if needed
        # Assuming keys are pocket paths
        test_items = []
        for pocket, ligand in test_data.items():
            test_items.append({"pocket": pocket, "ligand": ligand})
    elif isinstance(test_data, list):
        test_items = test_data
    else:
        print(f"Unknown data format: {type(test_data)}")
        return
        
    print(f"Loaded {len(test_items)} test cases.")
    
    # 2. Setup Results Directory
    if args.output_dir:
        benchmark_run_dir = args.output_dir
        timestamp = time.strftime("%Y%m%d_%H%M%S")
    else:
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "benchmark_results")
        os.makedirs(results_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        benchmark_run_dir = os.path.join(results_dir, f"run_{timestamp}")
    
    os.makedirs(benchmark_run_dir, exist_ok=True)
    
    summary_csv_path = os.path.join(benchmark_run_dir, "benchmark_summary.csv")
    
    # Check for resume
    completed_indices = set()
    all_results = []
    all_filtered_results = [] # New: Store Top N molecules per case
    
    # Load filtered summary if resuming
    filtered_summary_path = os.path.join(benchmark_run_dir, "benchmark_filtered_summary.csv")
    if os.path.exists(filtered_summary_path):
         try:
            df_filtered = pd.read_csv(filtered_summary_path)
            all_filtered_results = df_filtered.to_dict('records')
            print(f"Resuming: Loaded {len(all_filtered_results)} filtered molecules.")
         except:
            pass

    if os.path.exists(summary_csv_path):
        print(f"Found existing summary at {summary_csv_path}, attempting to resume...")
        try:
            df_existing = pd.read_csv(summary_csv_path)
            all_results = df_existing.to_dict('records')
            if "Case_ID" in df_existing.columns:
                completed_indices = set(df_existing["Case_ID"].unique())
                print(f"Resuming: {len(completed_indices)} cases already completed.")
            else:
                print("Warning: Existing summary missing 'Case_ID', cannot resume reliably.")
        except Exception as e:
            print(f"Error reading existing summary: {e}")

    # Redirect stdout/stderr to a global log file for this benchmark run
    global_log_path = os.path.join(benchmark_run_dir, "benchmark_run.log")
    
    print(f"Results and logs will be saved to: {benchmark_run_dir}")
    
    # We will also capture per-case logs if possible, but for now let's just log stdout
    class Logger(object):
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "a")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = Logger(global_log_path)
    # sys.stderr = Logger(global_log_path) # Optional: Redirect stderr too
    
    # 3. Main Loop
    # all_results is initialized above (empty or loaded from resume)
    
    # Limit to 1 for testing workflow
    # max_items = 1
    # test_items = test_items[:max_items]
    
    for i, item in enumerate(tqdm(test_items, desc="Benchmarking")):
        if i in completed_indices:
            continue
        pocket_path = item.get("pocket_path") or item.get("pocket")
        ligand_path = item.get("ligand_path") or item.get("ligand")
        
        if not pocket_path:
            print(f"Skipping item {i}: No pocket path")
            continue
            
        print(f"\n\n--- Running Case {i+1}/{len(test_items)} ---")
        print(f"Pocket: {pocket_path}")
        
        # Construct Prompt
        # User requirements:
        # 1. Shield ReportAgent (Do NOT generate report)
        # 2. Disable IntentAgent inquiry (Do NOT ask for clarification)
        # 3. Generate + Filter mode (Dynamic num_samples, QED>0.6, SA<4, Docking<-4)
        # 4. No Retrosynthesis
        # 5. Authentic workflow (Interact with IntentAgent)
        
        task_spec = {
            "task": "generation",
            "target_pdb": pocket_path,
            "reference_ligand": ligand_path,
            "num_molecules": args.num_samples,
            "requirements": {
                "strategy": "Generate then Filter",
                "filters": {
                    "QED": ">0.6",
                    "SA": "<4",
                    "Docking_Score": "<-4"
                }
            },
            "workflow": {
                "max_rounds": 3,
                "min_qualified": args.num_samples,
                "execution_mode": "authentic"
            },
            "restrictions": [
                "No retrosynthesis",
                "No PDF report",
                "No user clarification",
                "Avoid guided generation models (strictly enforce properties during generation)"
            ],
            "planning_rules": [
                "Plan must involve: TargetAgent -> GeneratorAgent -> EvaluatorAgent",
                "Include decision loop for insufficient qualified molecules"
            ]
        }
        
        prompt_content = json.dumps(task_spec, indent=2)

        # Initialize State
        run_id = f"bench_{timestamp}_{i}"
        initial_state = AgentState(
            messages=[],
            user_input=prompt_content,
            intent=None,
            task_params={},
            current_agent="IntentAgent",
            results={},
            error=None,
            is_complete=False,
            run_id=run_id,
            loop_count=0,
            plan=[],
            plan_step=0
        )
        
        start_time = time.time()
        
        # Init variables for safety
        eval_results = {}
        status = "Unknown"
        
        try:
            # Execute Workflow
            # We use the REAL app imported from interactive_workflow
            final_state = app.invoke(initial_state)
            
            duration = time.time() - start_time
            
            # Extract Results
            # We want to know:
            # - Did it finish successfully?
            # - All molecules generated and their metrics
            
            results = final_state.get("results", {})
            eval_results = results.get("evaluation", {})
            
            # Get Qualified Count for logging
            qualified_count = eval_results.get("qualified_count", 0)
            
            # Get Final Molecules (Top N + others)
            # Try to read the full evaluation CSV if available to get ALL molecules
            summary_csv = eval_results.get("summary_csv")
            if summary_csv and os.path.exists(summary_csv):
                try:
                    df_full = pd.read_csv(summary_csv)
                    final_mols = df_full.to_dict('records')
                except Exception as e:
                    print(f"Error reading full results from {summary_csv}: {e}")
                    final_mols = eval_results.get("final_molecules", [])
            else:
                final_mols = eval_results.get("final_molecules", [])
            
            status = "Success"
            if final_state.get("error"):
                status = f"Error: {final_state.get('error')}"
            elif final_state.get("intent") == "clarification_needed":
                status = "Failed: Clarification Requested"
            
            if final_mols:
                for mol in final_mols:
                    # Flatten molecule dict and add run-level info
                    row = mol.copy()
                    row["Pocket"] = pocket_path
                    row["Ligand"] = ligand_path
                    row["Status"] = status
                    row["Run_ID"] = run_id
                    row["Duration_s"] = round(duration, 2)
                    row["Case_ID"] = i # Track which case this belongs to
                    all_results.append(row)
            else:
                # No molecules generated, still log the failure
                all_results.append({
                    "Pocket": pocket_path,
                    "Ligand": ligand_path,
                    "Status": status,
                    "Run_ID": run_id,
                    "Case_ID": i
                })
            
            print(f"Case {i+1} Result: {status}, Qualified: {qualified_count}, Molecules: {len(final_mols)}")
            
        except Exception as e:
            print(f"Exception processing case {i}: {e}")
            traceback.print_exc()
            all_results.append({
                "Pocket": pocket_path,
                "Status": f"Exception: {str(e)}",
                "Run_ID": run_id,
                "Case_ID": i
            })
            
        # Incremental Save (Moved outside try/except block to ensure execution)
        # Save two summaries:
        # 1. Full Benchmark Summary (All Evaluated Molecules)
        pd.DataFrame(all_results).to_csv(summary_csv_path, index=False)
        
        # 2. Filtered Benchmark Summary (Top N Molecules per case)
        # Try to load the 'filtered_molecules.csv' from the run directory which contains the forced Top N
        filtered_csv = eval_results.get("feedback", {}).get("filtered_csv")
        print(f"[Benchmark] Filtered CSV path from Evaluator: {filtered_csv}")
        
        if filtered_csv and os.path.exists(filtered_csv):
            try:
                df_case_filtered = pd.read_csv(filtered_csv)
                case_filtered_mols = df_case_filtered.to_dict('records')
                print(f"[Benchmark] Loaded {len(case_filtered_mols)} filtered molecules for Case {i}")
                
                # Enrich with metadata
                for m in case_filtered_mols:
                    m["Pocket"] = pocket_path
                    m["Ligand"] = ligand_path
                    m["Status"] = status
                    m["Run_ID"] = run_id
                    m["Case_ID"] = i
                all_filtered_results.extend(case_filtered_mols)
                print(f"[Benchmark] Total accumulated filtered molecules: {len(all_filtered_results)}")
            except Exception as e:
                print(f"Error reading filtered csv {filtered_csv}: {e}")
        else:
            print(f"[Benchmark] Warning: Filtered CSV not found or None for Case {i}")
        
        filtered_summary_path = summary_csv_path.replace("benchmark_summary", "benchmark_filtered_summary")
        pd.DataFrame(all_filtered_results).to_csv(filtered_summary_path, index=False)
        
    print(f"\nBenchmark Complete. Summary saved to {summary_csv_path}")
    print(f"Filtered Summary saved to {filtered_summary_path}")

    # Final check of filtered summary count
    if all_filtered_results:
        print(f"Total Filtered Molecules: {len(all_filtered_results)} (Expected ~{len(test_items) * 20})")


if __name__ == "__main__":
    run_benchmark()
