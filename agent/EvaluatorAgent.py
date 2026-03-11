import os
import json
import numpy as np
import pandas as pd
import subprocess
import shutil
from typing import Dict, Any, List, Optional, Union
from langchain_core.messages import SystemMessage
from agent.base_agent import BaseAgent
from agent.state import AgentState
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, RDConfig, AllChem
import sys

# Add path for SA Score
# Using RDKit Contrib SA Score implementation
sa_score_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
if sa_score_path not in sys.path:
    sys.path.append(sa_score_path)

try:
    import sascorer
except ImportError:
    print("Warning: sascorer not found. SA Score calculation will fail.")
    sascorer = None

# Try importing TDC for ADMET
try:
    from tdc import Oracle
    TDC_AVAILABLE = True
except ImportError:
    TDC_AVAILABLE = False

class EvaluatorAgent(BaseAgent):
    def __init__(self):
        super().__init__(agent_name="EvaluatorAgent")
        self.work_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "evaluation_results")
        os.makedirs(self.work_dir, exist_ok=True)
        
        self.tool_implementations = {
            "CalculateQED": self.calculate_qed,
            "CalculateSA": self.calculate_sa,
            "CalculateLipinski": self.calculate_lipinski,
            "CalculateToxicity": self.calculate_toxicity,
            "CalculateDiversity": self.calculate_diversity,
            "RunStandardEvaluation": self.run_standard_evaluation,
            "RunDocking": self.run_docking,
            "ValidatePose": self.validate_pose,
            "AnalyzeInteractions": self.analyze_interactions,
            "FilterMolecules": self.filter_molecules,
            "Finish": None
        }
        
        self.tool_descriptions = {
            "CalculateQED": "Calculates Quantitative Estimate of Drug-likeness (QED). Input: 'molecules' (list of SMILES or SDF path). Output: Updated dataframe with QED column.",
            "CalculateSA": "Calculates Synthetic Accessibility (SA) score (1-10). Input: 'molecules'. Output: Updated dataframe with SA column.",
            "CalculateLipinski": "Calculates Lipinski's Rule of 5 properties (MW, LogP, HBD, HBA). Input: 'molecules'. Output: Updated dataframe.",
            "CalculateToxicity": "Calculates Toxicity/ADMET properties (Tox21, PAINS). Input: 'molecules'. Output: Updated dataframe with Toxicity columns.",
            "CalculateDiversity": "Calculates set-level metrics: Internal Diversity (IntDiv) and Uniqueness. Input: 'molecules'. Output: Returns metrics string and updates internal state.",
            "RunStandardEvaluation": "Runs a standard battery of tests: QED, SA, Lipinski, Toxicity, and Diversity. Recommended for initial screening. Input: 'molecules'. Output: Updated dataframe.",
            "RunDocking": "Runs molecular docking (Vina/Gnina). Input: 'molecules', 'target_pdb' (path), 'pocket_center' (list [x,y,z]). Output: Updated dataframe with Docking_Score column.",
            "ValidatePose": "Validates docked poses using PoseBusters. Input: 'molecules' (must be docked), 'target_pdb'. Output: Updated dataframe with PoseBusters_Pass column.",
            "AnalyzeInteractions": "Analyzes protein-ligand interactions using PLIP. Input: 'molecules' (must be docked), 'target_pdb'. Output: Updated dataframe with Interaction counts.",
            "FilterMolecules": "Filters molecules based on constraints. Input: 'constraints' (dict, e.g. {'QED': '>=0.6', 'Docking_Score': '<=-4.0'}). Output: Updates 'Qualified' status.",
            "Finish": "Call this when all requested evaluations are done. Input: 'status' ('sufficient' or 'insufficient'), 'missing_count' (int, optional), 'feedback' (dict, optional)."
        }
        
        # Internal state for the current evaluation session
        self.current_molecules_df = None
        self.current_run_dir = None
        
        # Persistent state for accumulation across loops
        self.all_molecules_df = pd.DataFrame()
        self.last_run_id = None

    def _compute_num_atoms(self, smiles_value: Any) -> Union[int, float]:
        if not isinstance(smiles_value, str) or not smiles_value:
            return np.nan
        mol = Chem.MolFromSmiles(smiles_value)
        if not mol:
            return np.nan
        return int(mol.GetNumAtoms())

    def _ensure_num_atoms_column(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty or "smiles" not in df.columns:
            return df
        df["num_atoms"] = [self._compute_num_atoms(s) for s in df["smiles"].tolist()]
        return df

    def _get_kg_tools(self) -> List[Dict]:
        return self.kg_loader.query_agent_tools("EvaluatorAgent")

    def run(self, state: AgentState) -> Dict[str, Any]:
        task_params = state.get("task_params", {})
        results = state.get("results", {})
        kg_tools = self._get_kg_tools()
        
        # Ensure evaluation results dict exists
        if "evaluation" not in results:
            results["evaluation"] = {}
            
        generation_results = results.get("generation", {})
        target_data = results.get("target_preparation", {})
        run_id = state.get("run_id", "default_run")
        
        # Create run-specific directory
        run_dir = os.path.join(self.work_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        self.current_run_dir = run_dir
        
        # Reset accumulation if new run
        if self.last_run_id != run_id:
            self.all_molecules_df = pd.DataFrame()
            self.last_run_id = run_id
            
        # 0. Check for molecules in task_params (Standalone Mode)
        # If generation_results is empty, we check if the user provided molecules directly in task_params.
        if not generation_results and ("molecules" in task_params or "smiles" in task_params):
             print("[EvaluatorAgent] No generation results, checking task_params for molecules...")
             mols = task_params.get("molecules") or task_params.get("smiles")
             if isinstance(mols, list):
                 # Construct a fake generation result for _load_molecules
                 generation_results = {"manual_input": {"smiles": mols}}
             elif isinstance(mols, str):
                 # It might be a single smile or comma-separated
                 if "," in mols:
                     generation_results = {"manual_input": {"smiles": [s.strip() for s in mols.split(",")]}}
                 else:
                     generation_results = {"manual_input": {"smiles": [mols]}}
            
        # 1. Load Molecules
        self.current_molecules_df = self._load_molecules(generation_results)
        if self.current_molecules_df is None or self.current_molecules_df.empty:
            print("[EvaluatorAgent] No molecules found to evaluate.")
            # Even if current batch is empty, we might have previous ones.
            # But usually this means generation failed.
            return {
                "current_agent": "EvaluatorAgent",
                "results": results,
                "task_params": task_params
            }
            
        print(f"[EvaluatorAgent] Loaded {len(self.current_molecules_df)} molecules for evaluation.")

        # --- IMMEDIATE ACCUMULATION LOGIC ---
        # Load previous results from final_evaluation.csv (if exists) and combine with current batch.
        # This ensures we have a complete picture BEFORE starting the evaluation loop.
        accumulated_csv_path = os.path.join(self.current_run_dir, "final_evaluation.csv")
        existing_df = pd.DataFrame()
        
        if os.path.exists(accumulated_csv_path):
            try:
                existing_df = pd.read_csv(accumulated_csv_path)
                print(f"[EvaluatorAgent] Loaded {len(existing_df)} existing molecules from {accumulated_csv_path}")
            except Exception as e:
                print(f"[EvaluatorAgent] Warning: Failed to read existing CSV: {e}")

        # Combine: Existing + Current
        dfs_to_concat = []
        if not existing_df.empty:
            dfs_to_concat.append(existing_df)
        if not self.current_molecules_df.empty:
            dfs_to_concat.append(self.current_molecules_df)
            
        if dfs_to_concat:
            self.all_molecules_df = pd.concat(dfs_to_concat, ignore_index=True)
        else:
            self.all_molecules_df = pd.DataFrame()
            
        # Deduplicate
        if "smiles" in self.all_molecules_df.columns and not self.all_molecules_df.empty:
            before_count = len(self.all_molecules_df)
            self.all_molecules_df.drop_duplicates(subset=["smiles"], inplace=True, keep="last")
            print(f"[EvaluatorAgent] Accumulation: {before_count} -> {len(self.all_molecules_df)} molecules (after deduplication).")
            
        # IMMEDIATE PERSISTENCE
        # REMOVED per user request: Do NOT save immediately to avoid writing placeholder data (SMILES only)
        # We will save only after evaluation in FilterMolecules or at the end of the run.
        # try:
        #    self.all_molecules_df.to_csv(accumulated_csv_path, index=False)
        #    print(f"[EvaluatorAgent] Persisted combined state to {accumulated_csv_path}")
        # except Exception as e:
        #     print(f"[EvaluatorAgent] Error saving combined CSV: {e}")
        
        # Add target info to context
        case_id = task_params.get("case_id") or task_params.get("target_id")
        context_data = {
            "target_pdb": target_data.get("cleaned_pdb") or target_data.get("original_pdb"),
            "pocket_center": target_data.get("pocket_center"),
            "molecule_count": len(self.current_molecules_df),
            "columns": list(self.current_molecules_df.columns),
            "case_id": case_id
        }
        
        # --- Persistence of Evaluation Goal & Loop Count ---
        # To prevent 'required_count' from being overwritten by partial batch sizes in loop (e.g., missing_count),
        # we persist the ORIGINAL target count for this run_id.
        config_path = os.path.join(self.current_run_dir, "evaluator_config.json")
        current_request_count = task_params.get("num_samples") or task_params.get("num_molecules") or 20
        try:
            current_request_count = int(current_request_count)
        except:
            current_request_count = 20
            
        required_count = current_request_count
        loop_count = 0
        
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                    persisted_count = config.get("required_count")
                    if persisted_count:
                        print(f"[EvaluatorAgent] Loaded persisted target count: {persisted_count} (Current request says: {current_request_count})")
                        required_count = int(persisted_count)
                    
                    loop_count = config.get("loop_count", 0)
                    print(f"[EvaluatorAgent] Loaded global loop count: {loop_count}")
            except Exception as e:
                print(f"[EvaluatorAgent] Warning: Failed to load evaluator config: {e}")
        else:
            # Initialize config
            try:
                with open(config_path, "w") as f:
                    json.dump({"required_count": required_count, "loop_count": 0}, f)
                print(f"[EvaluatorAgent] Persisted target count {required_count} to {config_path}")
            except Exception as e:
                print(f"[EvaluatorAgent] Warning: Failed to save evaluator config: {e}")
                
        context_data["loop_count"] = loop_count
        history = []
        max_steps = int(os.getenv("AGENT_MAX_STEPS", 10))
        step = 0
        
        print(f"[EvaluatorAgent] Starting ReAct loop (Max Steps: {max_steps})...")
        
        next_agent = "SynthesisAgent" # Default
        feedback = {}
        
        while step < max_steps:
            step += 1
            
            # A. Construct Prompt
            # Pass full accumulated history to prompt context, so LLM sees global progress
            current_accumulated_count = len(self.all_molecules_df) + len(self.current_molecules_df) if self.all_molecules_df.empty else len(self.all_molecules_df)
            
            # Update context_data to reflect the CURRENT batch being evaluated, but also hint at global state
            context_data["accumulated_count"] = current_accumulated_count
            
            prompt = self._construct_react_prompt(task_params, kg_tools, history, context_data)
            
            # B. Call LLM
            messages = [SystemMessage(content=prompt)]
            response = self.model.invoke(messages)
            content = response.content.strip()
            
            # C. Parse Action
            action = self._parse_action(content)
            if not action:
                print(f"[EvaluatorAgent] Failed to parse action: {content}")
                history.append({"step": step, "error": "Parse Error", "content": content})
                continue
                
            tool_name = action.get("tool")
            args = action.get("args", {})
            thought = action.get("thought", "")
            
            print(f"[EvaluatorAgent] Step {step}")
            print(f"  Thought: {thought}")
            print(f"  Tool: {tool_name} Args: {args}")
            
            if tool_name == "Finish":
                status = args.get("status", "sufficient")
                missing = args.get("missing_count", 0)
                
                if status == "insufficient":
                    next_agent = "GeneratorAgent"
                    feedback = {"missing_count": missing, "status": "insufficient"}
                else:
                    next_agent = "SynthesisAgent"
                    feedback = {"status": "sufficient"}
                break
                
            # D. Execute Tool
            result = self._execute_tool(tool_name, args, context_data)
            
            # Update context
            context_data["columns"] = list(self.current_molecules_df.columns)
            context_data["molecule_count"] = len(self.current_molecules_df)
            
            history.append({
                "step": step,
                "thought": thought,
                "tool": tool_name,
                "args": args,
                "result": str(result)
            })

        # 3. Save and Accumulate Results
        # Cleanup: Remove empty 'scores' column if present to avoid confusion
        if "scores" in self.current_molecules_df.columns:
            # Check if it's empty or all null (handling both NaN and empty strings)
            is_null = self.current_molecules_df["scores"].isnull()
            is_empty = self.current_molecules_df["scores"] == ""
            if (is_null | is_empty).all():
                self.current_molecules_df.drop(columns=["scores"], inplace=True)

        # Cleanup: Remove internal path columns not needed for final report
        cols_to_drop = [col for col in ["Docked_Pose_Path", "Docked_File"] if col in self.current_molecules_df.columns]
        if cols_to_drop:
            self.current_molecules_df.drop(columns=cols_to_drop, inplace=True)

        # --- Feedback Loop Logic (Now driven by LLM via 'Finish' tool) ---
        # The LLM has already decided 'next_agent' and 'feedback' in the ReAct loop.
        # But we need to enforce quality control and loop back if needed.
        
        qualified_count = 0
        constraints = task_params.get("constraints", {})
        
        # 1. Identify Qualified Molecules
        # The LLM should have already applied filtering via 'FilterMolecules' tool.
        # We trust the 'Qualified' column in the accumulated dataframe.
        
        # --- Persistence of Evaluation Goal ---
        # (Already handled at start of run)

            
        # --- Cumulative Filtering Logic ---
        # We filter from the FULL accumulated dataframe (self.all_molecules_df), not just the current batch.
        df = self.all_molecules_df.copy()
        
        # --- Filter by Case_ID if present (to ensure we count only current target) ---
        if case_id:
             col_name = None
             for c in ["Case_ID", "case_id", "target_id", "Target_ID"]:
                if c in df.columns:
                    col_name = c
                    break
             if col_name:
                 before_filter = len(df)
                 df = df[df[col_name].astype(str) == str(case_id)]
                 print(f"[EvaluatorAgent] Final Accumulation Check: Filtered by {col_name}={case_id}, count: {before_filter} -> {len(df)}")
        
        if "Qualified" not in df.columns:
            # If LLM didn't filter, assume all are qualified (or fallback to strict default?)
            # Let's assume True to avoid blocking flow, but log warning.
            print("[EvaluatorAgent] Warning: 'Qualified' column missing. Assuming all molecules are qualified.")
            df["Qualified"] = True 

        # Ensure Qualified is boolean
        df["Qualified"] = df["Qualified"].astype(bool)
        
        qualified_df = df[df["Qualified"] == True]
        qualified_count = len(qualified_df)
        print(f"[EvaluatorAgent] Total Qualified molecules found (Accumulated): {qualified_count}/{required_count}")

        # 2. Decide Next Step based on ACCUMULATED count
        if qualified_count < required_count:
            next_agent = "GeneratorAgent"
            missing = required_count - qualified_count
            
            # --- LOOP PREVENTION LOGIC & MINIMUM BATCH SIZE ---
            # Force minimum request of 10
            request_count = missing
            if request_count < 10:
                print(f"[EvaluatorAgent] Missing count {missing} is too low. Forcing minimum request of 10.")
                request_count = 10
            
            feedback = {
                "status": "insufficient",
                "qualified_count": qualified_count,
                "missing_count": missing,
                "num_molecules": request_count,
                "feedback": f"Accumulated only {qualified_count} qualified molecules. Need {missing} more. Requesting {request_count}."
            }
            print(f"[EvaluatorAgent] Insufficient qualified molecules ({qualified_count}/{required_count}). Requesting {request_count} more.")
        else:
            next_agent = "SynthesisAgent" # Or Finish
            feedback = {"status": "sufficient", "qualified_count": qualified_count}
            
        # --- CRITICAL FIX: Ensure 'Qualified' status is calculated for the FULL dataset ---
        # We use 'df' which is already filtered by Case_ID and has Qualified column handled.
        
        if "Qualified" in df.columns:
             # Fill NaN with False to be safe
             df["Qualified"] = df["Qualified"].fillna(False)
             # Re-calculate qualified_df
             qualified_df = df[df["Qualified"] == True]
             qualified_count = len(qualified_df)
             
             # Re-evaluate sufficiency
             if qualified_count >= required_count:
                 next_agent = "SynthesisAgent"
                 feedback = {"status": "sufficient", "qualified_count": qualified_count}
             else:
                 # Check Loop Limit
                 max_loops = int(os.getenv("EVALUATOR_MAX_LOOPS", 3))
                 if loop_count >= max_loops:
                     print(f"[EvaluatorAgent] Max loops ({max_loops}) reached. Forcing selection of top {required_count} molecules.")
                     next_agent = "SynthesisAgent"
                     feedback = {
                         "status": "sufficient", 
                         "qualified_count": qualified_count,
                         "note": "Forced selection due to max loops"
                     }
                 else:
                     # Update feedback with corrected count
                     missing = required_count - qualified_count
                     request_count = missing
                     if request_count < 10:
                         request_count = 10
                     
                     feedback["missing_count"] = missing
                     feedback["num_molecules"] = request_count
                     feedback["qualified_count"] = qualified_count
                     feedback["feedback"] = f"Accumulated only {qualified_count} qualified molecules. Need {missing} more. Requesting {request_count}."

        # 3. Sort and Select Top Molecules for Final Output (Re-ranking)
        # We re-rank the ENTIRE pool every time.
        sort_reason = None
        # Prefer MPO-based ranking
        if {"Docking_Score", "QED", "SA"}.issubset(set(df.columns)):
            try:
                df = self._compute_mpo(df)
                df = df.sort_values(by=["Qualified", "MPO_Score"], ascending=[False, False])
                sort_reason = "Qualified + MPO_Score"
            except:
                pass
                
        if sort_reason is None:
             if "Docking_Score" in df.columns:
                df = df.sort_values(by=["Qualified", "Docking_Score"], ascending=[False, True]) # Lower docking is better
                sort_reason = "Qualified + Docking_Score"
             elif "QED" in df.columns:
                df = df.sort_values(by=["Qualified", "QED"], ascending=[False, False])
                sort_reason = "Qualified + QED"
             else:
                sort_reason = "Qualified + Original"

        # Select Top N for downstream (filtered_molecules.csv)
        # Always take the absolute best N from the accumulated history.
        print(f"[EvaluatorAgent] Filtering Logic: Required={required_count}, Total Accumulated={len(df)}, Qualified={len(qualified_df)}")
        
        top_molecules = df.head(required_count).copy() # Work on a copy
        
        filtered_csv = os.path.join(self.current_run_dir, "filtered_molecules.csv")
        
        top_molecules = self._ensure_num_atoms_column(top_molecules)

        if "rank" in top_molecules.columns:
            top_molecules.drop(columns=["rank"], inplace=True)
        top_molecules.insert(0, "rank", range(1, len(top_molecules) + 1))

        if "source" in top_molecules.columns:
            top_molecules.drop(columns=["source"], inplace=True)
            
        top_molecules.to_csv(filtered_csv, index=False)
        feedback["filtered_csv"] = filtered_csv
        results["evaluation"]["final_molecules"] = top_molecules.to_dict(orient="records")
        print(f"[EvaluatorAgent] Top {len(top_molecules)} molecules (from accumulated history) sorted by {sort_reason} saved to {filtered_csv}.")
        
        # Update global state
        if "evaluation" not in results:
            results["evaluation"] = {}
            
        # --- CRITICAL: Persist the FULL accumulated dataframe to disk for next loop ---
        # This ensures that next time we run, we can load everything back.
        try:
            self.all_molecules_df = self._ensure_num_atoms_column(self.all_molecules_df)
            self.all_molecules_df.to_csv(accumulated_csv_path, index=False)
            print(f"[EvaluatorAgent] Persisted all {len(self.all_molecules_df)} molecules to {accumulated_csv_path}")
        except Exception as e:
            print(f"[EvaluatorAgent] Error saving accumulated CSV: {e}")
            
        results["evaluation"]["summary_csv"] = accumulated_csv_path
        results["evaluation"]["top_molecules"] = self.all_molecules_df.head(5).to_dict(orient="records")
        results["evaluation"]["valid_count"] = len(self.all_molecules_df)
        results["evaluation"]["qualified_count"] = qualified_count
        results["evaluation"]["next_agent"] = next_agent
        
        # --- Update Loop Count if looping ---
        # Must update BEFORE putting into feedback
        if next_agent == "GeneratorAgent":
            loop_count += 1
            try:
                with open(config_path, "w") as f:
                    json.dump({"required_count": required_count, "loop_count": loop_count}, f)
                print(f"[EvaluatorAgent] Updated global loop count to {loop_count}")
            except Exception as e:
                print(f"[EvaluatorAgent] Warning: Failed to update evaluator config: {e}")
        
        # Add loop info to feedback for visibility (AFTER update)
        feedback["loop_count"] = loop_count
        results["evaluation"]["feedback"] = feedback
        results["evaluation"]["loop_count"] = loop_count # Also at top level for safety
        
        return {
            "current_agent": "EvaluatorAgent",
            "results": results,
            "task_params": task_params
        }

    def _load_molecules(self, generation_results: Dict) -> pd.DataFrame:
        """Parses generation results into a DataFrame."""
        all_mols = []
        
        for tool, res in generation_results.items():
            if isinstance(res, dict):
                # Handle different response formats
                # Case 1: List of SMILES
                if "smiles" in res:
                    for smi in res["smiles"]:
                        if smi: all_mols.append({"smiles": smi, "source": tool})
                # Case 2: SDF file path
                elif "sdf_path" in res:
                    sdf_path = res["sdf_path"]
                    if os.path.exists(sdf_path):
                        suppl = Chem.SDMolSupplier(sdf_path)
                        for mol in suppl:
                            if mol:
                                smi = Chem.MolToSmiles(mol)
                                if smi: all_mols.append({"smiles": smi, "source": tool, "mol_obj": mol})
                # Case 3: List of objects (DiffSBDD, DiffGui, GenMol)
                elif "molecules" in res:
                    for m in res["molecules"]:
                        if isinstance(m, str):
                            if m: all_mols.append({"smiles": m, "source": tool})
                        elif isinstance(m, dict) and "smiles" in m:
                            # Create a copy to avoid modifying the original dict in place if needed
                            mol_entry = m.copy()
                            mol_entry["source"] = tool
                            if mol_entry["smiles"]: all_mols.append(mol_entry)
                            
        if not all_mols:
            return None
            
        return pd.DataFrame(all_mols)

    def _construct_react_prompt(self, task_params, kg_tools, history, context_data):
        # Filter tools based on context (e.g., skip docking if no target)
        available_tools = self.tool_descriptions.copy()
        target_pdb = context_data.get('target_pdb')
        
        if not target_pdb:
            # Remove target-dependent tools
            for tool in ["RunDocking", "ValidatePose", "AnalyzeInteractions"]:
                if tool in available_tools:
                    del available_tools[tool]
        
        tools_desc = json.dumps(available_tools, indent=2)
        
        # Format KG tools for context (Reasoning Layer)
        kg_context_lines = []
        for t in kg_tools:
            name = t.get('name', 'Unknown')
            func = t.get('function', '')
            cat = t.get('category', '')
            kg_context_lines.append(f"- {name} ({cat}): {func}")
        kg_context_str = "\n".join(kg_context_lines)

        history_lines = []
        for h in history:
            if "error" in h:
                history_lines.append(f"Step {h['step']}: Error - {h['error']} (Content: {h.get('content', '')[:100]}...)")
            else:
                history_lines.append(f"Step {h['step']}: {h.get('thought', '')} -> {h.get('tool', 'Unknown')}({h.get('args', {})}) -> {h.get('result', '')}")
        history_str = "\n".join(history_lines)

        task_params_for_prompt = dict(task_params) if isinstance(task_params, dict) else {"task_params": task_params}
        if "agent_logs" in task_params_for_prompt:
            task_params_for_prompt = {k: v for k, v in task_params_for_prompt.items() if k != "agent_logs"}
        
        prompt = f"""
You are the EvaluatorAgent. Your goal is to evaluate generated molecules using various metrics and tools.

### Task Parameters:
{json.dumps(task_params_for_prompt, indent=2)}

### Current Context:
- Target PDB: {context_data.get('target_pdb')}
- Pocket Center: {context_data.get('pocket_center')}
- Molecules Count: {context_data.get('molecule_count')}
- Current Data Columns: {context_data.get('columns')}

### Knowledge Graph Insights (Domain Capabilities):
The Knowledge Graph defines the theoretical capabilities available to you. Use this for high-level planning.
{kg_context_str}

### Available Tools (Executable Actions):
These are the actual Python tools you can call. Note that one Executable Action might implement multiple KG capabilities (e.g., 'RunDocking' implements 'AutoDock Vina' and 'Gnina').
{tools_desc}

### In-Context Examples (Few-Shot Learning):
Example 1:
Task: "Evaluate the generated molecules for drug-likeness and safety."
Step 1:
{{
  "thought": "To evaluate drug-likeness and safety, I need to calculate QED, SA Score, Lipinski properties, and Toxicity alerts. I will start with QED.",
  "tool": "CalculateQED",
  "args": {{}}
}}
Step 2:
{{
  "thought": "QED is done. Now I will calculate Synthetic Accessibility (SA).",
  "tool": "CalculateSA",
  "args": {{}}
}}
Step 3:
{{
  "thought": "Next, I need Lipinski properties (MW, LogP, etc.).",
  "tool": "CalculateLipinski",
  "args": {{}}
}}
Step 4:
{{
  "thought": "Finally, I will check for Toxicity/PAINS alerts.",
  "tool": "CalculateToxicity",
  "args": {{}}
}}
Step 5:
{{
  "thought": "All requested properties are calculated. I can finish.",
  "tool": "Finish",
  "args": {{ "status": "sufficient" }}
}}

Example 2:
Task: "Evaluate binding affinity for the target."
Step 1:
{{
  "thought": "The KG lists AutoDock Vina and Gnina for Docking. I will use the RunDocking action which handles the docking process.",
  "tool": "RunDocking",
  "args": {{ "target_pdb": "{context_data.get('target_pdb')}", "pocket_center": {context_data.get('pocket_center')} }}
}}

### Execution History:
{history_str}

### Instructions:
1. **Plan**: Look at the 'Knowledge Graph Insights' to understand the nature of the requested metrics.
2. **Map**: Find the corresponding 'Executable Action' in 'Available Tools'.
3. **Act**: Generate the JSON for the tool call.
   - **Important**: The examples above are for format and logic demonstration only. Do not blindly copy the sequence. Adapt your plan dynamically based on the specific `Task Parameters` and `Current State`.
4. **Standard Operating Procedure (SOP)**:
   - **Step 1: Assessment**: You **MUST** ensure that the following properties are calculated for ALL molecules: **Drug-likeness (QED)**, **Synthesizability (SA)**, **Lipinski Rules**, **Toxicity (PAINS)**, and **Diversity**. Look at the 'Available Tools' to find the appropriate tools to calculate these specific properties.
   - **Step 2: Docking (CRITICAL)**: 
     - **IF** 'Target PDB' and 'Pocket Center' are available in the Context, you **MUST** run 'RunDocking'. This is **MANDATORY** for target-based tasks. Do NOT finish without it.
     - **IF** no target is provided (unconditional generation), you should SKIP docking.
   - **Step 3: Filtering (INTELLIGENT TRANSLATION REQUIRED)**: 
     - Analyze the `Task Parameters` to find constraints (e.g., "qed_min: 0.6", "sa_max: 4.0", "docking_score_max: -4").
     - You MUST translate these into the standardized format for `FilterMolecules` (e.g., `{{ "QED": ">=0.6", "SA": "<=4.0", "Docking_Score": "<=-4.0" }}`).
     - Call `FilterMolecules` with these translated constraints. This updates the 'Qualified' status of molecules.
   - **Step 4: Verification**: Check if the number of *qualified* molecules meets the user's `num_samples` requirement.
     - If `qualified < num_samples`: Call `Finish(status='insufficient', missing_count=num_samples-qualified)`.
     - If `qualified >= num_samples`: Call `Finish(status='sufficient')`.

### Output Format (JSON):
{{
  "thought": "Reasoning based on KG and Context...",
  "tool": "ToolName",
  "args": {{ "arg1": "value1" }}
}}
"""
        return prompt

    def _parse_action(self, content):
        try:
            # Remove potential C-style comments which are invalid in JSON
            import re
            content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
            
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            else:
                json_str = content
            return json.loads(json_str)
        except:
            return None

    def _execute_tool(self, tool_name, args, context_data):
        if tool_name not in self.tool_implementations:
            return f"Error: Tool {tool_name} not found."
            
        func = self.tool_implementations[tool_name]
        
        # Inject implicit arguments if needed
        if tool_name == "RunDocking":
            if "target_pdb" not in args:
                args["target_pdb"] = context_data.get("target_pdb")
            if "pocket_center" not in args:
                args["pocket_center"] = context_data.get("pocket_center")
                
        if tool_name == "FilterMolecules":
            if "case_id" not in args:
                args["case_id"] = context_data.get("case_id")
                
        try:
            result = func(**args)
            print(f"[EvaluatorAgent] Tool Result: {str(result)[:200]}...") # Print start of result
            return result
        except Exception as e:
            error_msg = f"Error executing {tool_name}: {str(e)}"
            print(f"[EvaluatorAgent] {error_msg}")
            return error_msg

    # --- Tool Implementations ---

    def calculate_qed(self, **kwargs):
        if "QED" in self.current_molecules_df.columns:
            return "QED already calculated."
            
        def _calc(row):
            mol = self._get_mol(row)
            return QED.qed(mol) if mol else 0.0
            
        self.current_molecules_df["QED"] = self.current_molecules_df.apply(_calc, axis=1)
        return f"Calculated QED for {len(self.current_molecules_df)} molecules."

    def calculate_sa(self, **kwargs):
        if sascorer is None:
            return "SA Score tool not available."
            
        if "SA" in self.current_molecules_df.columns:
            return "SA already calculated."
            
        def _calc(row):
            mol = self._get_mol(row)
            return sascorer.calculateScore(mol) if mol else 10.0
            
        self.current_molecules_df["SA"] = self.current_molecules_df.apply(_calc, axis=1)
        return f"Calculated SA Score for {len(self.current_molecules_df)} molecules."

    def calculate_lipinski(self, **kwargs):
        def _calc(row):
            mol = self._get_mol(row)
            if not mol: return pd.Series([0, 0, 0, 0])
            return pd.Series([
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol)
            ])
            
        self.current_molecules_df[["MW", "LogP", "HBD", "HBA"]] = self.current_molecules_df.apply(_calc, axis=1)
        return "Calculated Lipinski properties."

    def calculate_toxicity(self, **kwargs):
        """Calculates Toxicity/ADMET properties."""
        if "PAINS_Alerts" in self.current_molecules_df.columns:
            return "Toxicity already calculated."

        # 1. RDKit PAINS (Structural Alerts)
        from rdkit.Chem import FilterCatalog
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
        catalog = FilterCatalog.FilterCatalog(params)
        
        def _check_pains(row):
            mol = self._get_mol(row)
            if not mol: return False
            return catalog.HasMatch(mol)
            
        self.current_molecules_df["PAINS_Alerts"] = self.current_molecules_df.apply(_check_pains, axis=1)
        
        # 2. TDC Oracles (if available)
        msg = "Calculated PAINS Alerts."
        if TDC_AVAILABLE:
            try:
                # Use a lightweight oracle like GSK or QED (already done)
                # Tox21 is a multi-task dataset, might be heavy. 
                # Let's use 'medchem_filters' if available or just stick to PAINS for now to avoid huge downloads.
                # Actually, let's try 'drd2' or similar if relevant, but for general toxicity:
                # 'ld50_zhu' is available in TDC.
                pass 
            except Exception as e:
                print(f"TDC calculation failed: {e}")
                
        return msg

    def calculate_diversity(self, **kwargs):
        """Calculates Internal Diversity and Uniqueness."""
        from rdkit import DataStructs
        
        mols = [self._get_mol(row) for _, row in self.current_molecules_df.iterrows()]
        # Keep track of valid indices to assign back to DataFrame correctly
        valid_indices = [i for i, m in enumerate(mols) if m is not None]
        valid_mols = [m for m in mols if m is not None]
        
        if not valid_mols:
            return "No valid molecules to calculate diversity."
            
        # 1. Uniqueness
        smiles_list = [Chem.MolToSmiles(m) for m in valid_mols]
        unique_smiles = set(smiles_list)
        uniqueness = len(unique_smiles) / len(smiles_list) if smiles_list else 0.0
        
        # 2. Internal Diversity (IntDiv)
        from rdkit.Chem import rdFingerprintGenerator
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fps = [mfpgen.GetFingerprint(m) for m in valid_mols]

        if len(fps) < 2:
            int_div = 0.0
            mean_sims = [0.0] * len(fps)
        else:
            sims = []
            mean_sims = []
            for i in range(len(fps)):
                # Calculate similarity to all other molecules
                # Note: This includes self-similarity (1.0) if we used the full list, 
                # but BulkTanimotoSimilarity with slicing excludes the current one if we slice around it.
                # To be precise for "Mean Similarity to Others":
                others = fps[:i] + fps[i+1:]
                if others:
                    s_i = DataStructs.BulkTanimotoSimilarity(fps[i], others)
                    mean_sim = np.mean(s_i)
                else:
                    mean_sim = 0.0
                mean_sims.append(mean_sim)
                
                # For IntDiv (global), we usually take the average of all pairwise similarities (excluding self)
                sims.extend(s_i)
                
            avg_sim = np.mean(sims) if sims else 0.0
            int_div = 1.0 - avg_sim
            
        # Add metrics to DataFrame
        # Initialize columns with NaN or default
        self.current_molecules_df["Diversity_Score"] = 0.0
        self.current_molecules_df["Uniqueness_Flag"] = False
        
        # Assign values for valid molecules
        # Diversity_Score = 1 - Mean_Similarity (Higher is more diverse)
        div_scores = [1.0 - s for s in mean_sims]
        
        # Uniqueness Flag (is this SMILES unique in the set?)
        # We need to check counts
        from collections import Counter
        counts = Counter(smiles_list)
        unique_flags = [counts[s] == 1 for s in smiles_list]
        
        # Map back to original dataframe using valid_indices
        # This is a bit tricky if dataframe has reset index. 
        # Assuming current_molecules_df index aligns with iterrows order (0..N-1)
        # We can use iloc.
        
        for idx, score, is_unique in zip(valid_indices, div_scores, unique_flags):
            self.current_molecules_df.iloc[idx, self.current_molecules_df.columns.get_loc("Diversity_Score")] = score
            self.current_molecules_df.iloc[idx, self.current_molecules_df.columns.get_loc("Uniqueness_Flag")] = is_unique
            
        return f"Diversity Metrics: Uniqueness={uniqueness:.4f}, IntDiv={int_div:.4f}"

    def run_standard_evaluation(self, **kwargs):
        """Runs QED, SA, Lipinski, Toxicity, and Diversity."""
        res = []
        res.append(self.calculate_qed())
        res.append(self.calculate_sa())
        res.append(self.calculate_lipinski())
        res.append(self.calculate_toxicity())
        res.append(self.calculate_diversity())
        return "Standard evaluation complete: " + ", ".join(res)

    def run_docking(self, target_pdb: str, pocket_center: List[float], **kwargs):
        if "Docking_Score" in self.current_molecules_df.columns:
             # Check if we have valid scores (non-null)
             if not self.current_molecules_df["Docking_Score"].isnull().all():
                 return "Docking scores already present."

        if not target_pdb or not os.path.exists(target_pdb):
            return "Error: Target PDB not found."
        if not pocket_center:
            return "Error: Pocket center not defined."
            
        # Check for Vina/Gnina
        # Strategy: Prioritize Vina for speed, fallback to Gnina (CPU) if Vina missing or fails
        docking_tool = None
        fallback_tool = None
        
        # Check Vina
        if shutil.which("vina"):
             docking_tool = "vina"
        
        # Check Gnina
        local_gnina = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools", "gnina")
        gnina_path = None
        if os.path.exists(local_gnina):
             gnina_path = local_gnina
             os.chmod(local_gnina, 0o755)
        elif shutil.which("gnina"): 
             gnina_path = "gnina"
        
        # Set tools
        if docking_tool:
            if gnina_path:
                fallback_tool = gnina_path
        elif gnina_path:
            docking_tool = gnina_path
        else:
             return "Error: No docking tool (vina/gnina) found."

        print(f"[EvaluatorAgent] Using {docking_tool} for docking. (Fallback: {fallback_tool})")
        
        # Prepare Receptor (PDB -> PDBQT)
        receptor_pdbqt = target_pdb.replace(".pdb", ".pdbqt")
        if not os.path.exists(receptor_pdbqt):
            self._prepare_receptor(target_pdb, receptor_pdbqt)
            
        # --- Parallel Execution Setup ---
        # from concurrent.futures import ThreadPoolExecutor, as_completed # Removed parallel execution
        from dotenv import load_dotenv
        
        # Load env for concurrency settings
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        load_dotenv(env_path, override=True)
        
        # We ignore DOCKING_CONCURRENCY as we are reverting to serial execution
        # But we ensure DOCKING_CPU_CORES is set correctly in .env (should be 24)
            
        print(f"[EvaluatorAgent] Starting docking for {len(self.current_molecules_df)} molecules sequentially (CPU Cores per task: {os.getenv('DOCKING_CPU_CORES', 'Unknown')})...")
        
        # Results container
        scores = []
        docked_files = []

        for idx, row in self.current_molecules_df.iterrows():
            try:
                mol = self._get_mol(row)
                if not mol:
                    scores.append(0.0)
                    docked_files.append(None)
                    continue
                    
                # Prepare Ligand
                ligand_name = f"mol_{idx}"
                ligand_pdbqt = os.path.join(self.current_run_dir, f"{ligand_name}.pdbqt")
                self._prepare_ligand(mol, ligand_pdbqt)
                
                if not os.path.exists(ligand_pdbqt):
                    print(f"[EvaluatorAgent] Failed to prepare ligand {idx}")
                    scores.append(0.0)
                    docked_files.append(None)
                    continue
                    
                # Run Docking
                score = self._run_single_docking(docking_tool, receptor_pdbqt, ligand_pdbqt, pocket_center)
                
                # RETRY / FALLBACK LOGIC
                if score == 0.0:
                    # 1. Fallback to Gnina if Vina failed
                    if fallback_tool and docking_tool != fallback_tool:
                        # print(f"[EvaluatorAgent] Docking failed with {docking_tool}. Retrying with {fallback_tool}...")
                        score = self._run_single_docking(fallback_tool, receptor_pdbqt, ligand_pdbqt, pocket_center)
                    
                    # 2. If still failing, try OpenBabel prep
                    current_tool = fallback_tool if (fallback_tool and score == 0.0) else docking_tool
                    
                    if score == 0.0:
                        # print(f"[EvaluatorAgent] Docking failed. Retrying with OpenBabel preparation (Attempt 1/1)...")
                        self._prepare_ligand(mol, ligand_pdbqt, use_openbabel=True)
                        score = self._run_single_docking(current_tool, receptor_pdbqt, ligand_pdbqt, pocket_center)
                        
                        if score == 0.0:
                            pass # Still failed
                
                # Output path check
                out_file = ligand_pdbqt.replace(".pdbqt", "_out.pdbqt")
                final_path = out_file if os.path.exists(out_file) else None
                
                scores.append(score)
                docked_files.append(final_path)
                
                if (idx + 1) % 5 == 0:
                     print(f"[EvaluatorAgent] Progress: {idx + 1}/{len(self.current_molecules_df)} molecules docked.")
                     
            except Exception as e:
                # Catch exceptions to prevent loop crash
                print(f"[EvaluatorAgent] Exception during docking for molecule {idx}: {str(e)}")
                scores.append(0.0)
                docked_files.append(None)

        print(f"[EvaluatorAgent] Docking loop finished. Scores count: {len(scores)}")
        self.current_molecules_df["Docking_Score"] = scores
        self.current_molecules_df["Docked_Pose_Path"] = docked_files
        print(f"[EvaluatorAgent] Columns added. Current columns: {self.current_molecules_df.columns.tolist()}")
        return f"Docking complete. Average Docking Score: {np.mean(scores):.2f}"

    def validate_pose(self, target_pdb: str, **kwargs):
        if "Docked_Pose_Path" not in self.current_molecules_df.columns:
            return "Error: Molecules must be docked first (run RunDocking)."
            
        try:
            import posebusters
        except ImportError:
            return "Error: PoseBusters not installed."

        results = []
        for idx, row in self.current_molecules_df.iterrows():
            docked_file = row.get("Docked_Pose_Path")
            if not docked_file or not os.path.exists(docked_file):
                results.append(False)
                continue
            
            # PoseBusters expects SDF/PDB usually. PDBQT might be an issue.
            # Let's convert PDBQT output to SDF for PoseBusters
            sdf_file = docked_file.replace(".pdbqt", ".sdf")
            subprocess.run(["obabel", docked_file, "-O", sdf_file], check=False)
            
            try:
                # Run PoseBusters on single file
                buster = posebusters.PoseBusters(config="mol")
                df_res = buster.bust([sdf_file], target_pdb, full_report=False)
                # Check if it passed all tests (or specific ones)
                # 'all_tests_passed' is a common column if available
                passed = True 
                if not df_res.empty and "all_tests_passed" in df_res.columns:
                     passed = bool(df_res["all_tests_passed"].iloc[0])
                results.append(passed)
            except Exception as e:
                print(f"PoseBusters failed for {docked_file}: {e}")
                results.append(False)

        self.current_molecules_df["PoseBusters_Pass"] = results
        return f"PoseBusters validation complete. Passing: {sum(results)}/{len(results)}"

    def analyze_interactions(self, target_pdb: str, **kwargs):
        if "Docked_Pose_Path" not in self.current_molecules_df.columns:
            return "Error: Molecules must be docked first (run RunDocking)."
            
        try:
            from plip.structure.preparation import PDBComplex
            from plip.exchange.report import BindingSiteReport
        except ImportError:
            return "Error: PLIP not installed."

        interactions = []
        for idx, row in self.current_molecules_df.iterrows():
            docked_file = row.get("Docked_Pose_Path")
            if not docked_file or not os.path.exists(docked_file):
                interactions.append({})
                continue
                
            # PLIP needs a complex PDB.
            # 1. Convert Ligand PDBQT -> PDB
            ligand_pdb = docked_file.replace(".pdbqt", ".pdb")
            subprocess.run(["obabel", docked_file, "-O", ligand_pdb], check=False)
            
            # 2. Concatenate Receptor + Ligand
            complex_pdb = docked_file.replace(".pdbqt", "_complex.pdb")
            with open(target_pdb, 'r') as f_rec, open(ligand_pdb, 'r') as f_lig, open(complex_pdb, 'w') as f_out:
                # Write receptor (excluding END/CONECT if needed, but simple cat usually works for PLIP)
                for line in f_rec:
                    if not line.startswith("END") and not line.startswith("CONECT"):
                        f_out.write(line)
                # Write ligand
                for line in f_lig:
                    f_out.write(line)
                f_out.write("END\n")
                
            try:
                # Run PLIP
                mol_complex = PDBComplex()
                mol_complex.load_pdb(complex_pdb)
                mol_complex.analyze()
                
                # Extract interactions
                # PLIP stores results in mol_complex.interaction_sets (dict by ligand)
                # Since we have one ligand, we take the first one
                if mol_complex.interaction_sets:
                    lig_key = list(mol_complex.interaction_sets.keys())[0]
                    iset = mol_complex.interaction_sets[lig_key]
                    
                    summary = {
                        "H-Bonds": len(iset.hbonds_p) + len(iset.hbonds_l),
                        "Hydrophobic": len(iset.hydrophobic_contacts),
                        "Pi-Stacking": len(iset.pistacking),
                        "Salt Bridges": len(iset.saltbridge_l0) + len(iset.saltbridge_p0)
                    }
                    interactions.append(summary)
                else:
                    interactions.append({})
            except Exception as e:
                print(f"PLIP failed for {docked_file}: {e}")
                interactions.append({})

        # Add columns
        hbonds = [i.get("H-Bonds", 0) for i in interactions]
        hydrophobic = [i.get("Hydrophobic", 0) for i in interactions]
        
        self.current_molecules_df["H_Bonds"] = hbonds
        self.current_molecules_df["Hydrophobic_Contacts"] = hydrophobic
        
        return f"PLIP analysis complete. Avg H-Bonds: {np.mean(hbonds):.1f}"

    def filter_molecules(self, constraints: Dict[str, str], case_id: str = None, **kwargs):
        """
        Filters molecules based on provided constraints and updates the 'Qualified' column.
        Reads from and writes to final_evaluation.csv in the current run directory.
        """
        # 1. Determine summary file path
        # We ONLY use final_evaluation.csv in the current run directory.
        summary_path = os.path.join(self.current_run_dir, "final_evaluation.csv")
            
        print(f"[EvaluatorAgent] FilterMolecules: using summary file {summary_path}")
        
        # 2. Load Data and Append Current Batch
        # We start by loading the HISTORICAL data from disk (if any).
        combined_df = pd.DataFrame()
        
        if os.path.exists(summary_path):
             try:
                 combined_df = pd.read_csv(summary_path)
                 print(f"[EvaluatorAgent] Loaded {len(combined_df)} historical molecules from {summary_path}")
             except Exception as e:
                 print(f"[EvaluatorAgent] Warning: Failed to read summary file: {e}")
        
        # Now append the CURRENT batch (which should now have scores calculated)
        # We use self.current_molecules_df which is in memory.
        if not self.current_molecules_df.empty:
             print(f"[EvaluatorAgent] Appending {len(self.current_molecules_df)} current molecules to history.")
             combined_df = pd.concat([combined_df, self.current_molecules_df], ignore_index=True)
        else:
             print("[EvaluatorAgent] Warning: Current molecules dataframe is empty.")

        # Deduplicate (keep last to ensure latest scores are kept if re-evaluated)
        if "smiles" in combined_df.columns and not combined_df.empty:
            before_count = len(combined_df)
            combined_df.drop_duplicates(subset=["smiles"], inplace=True, keep="last")
            print(f"[EvaluatorAgent] Total accumulated: {before_count} -> {len(combined_df)} (after deduplication)")

        # 3. Filter by Target (Case_ID)
        # (Logic remains same: double check we are filtering the right target)
        if case_id:
            col_name = None
            for c in ["Case_ID", "case_id", "target_id", "Target_ID"]:
                if c in combined_df.columns:
                    col_name = c
                    break
            
            if col_name:
                initial_len = len(combined_df)
                combined_df = combined_df[combined_df[col_name].astype(str) == str(case_id)]
                print(f"[EvaluatorAgent] Filtered by {col_name}={case_id}: {initial_len} -> {len(combined_df)}")
            else:
                # If no Case_ID column, we assume all data in this run folder belongs to this case.
                pass

        # 4. Apply Constraints
        if "Qualified" not in combined_df.columns:
            combined_df["Qualified"] = True
            
        # Ensure Qualified is boolean and fill NaN with True (default)
        combined_df["Qualified"] = combined_df["Qualified"].fillna(True).astype(bool)

        applied_constraints = []
        
        # We need to apply constraints to the WHOLE dataframe to ensure consistency
        # But we only want to update the 'Qualified' column.
        
        for col, condition in constraints.items():
            if col not in combined_df.columns:
                print(f"[EvaluatorAgent] Warning: Column '{col}' not found for filtering.")
                continue
                
            try:
                condition = str(condition).strip()
                operator = ""
                value_str = ""
                
                if condition.startswith(">="):
                    operator = ">="
                    value_str = condition[2:]
                elif condition.startswith("<="):
                    operator = "<="
                    value_str = condition[2:]
                elif condition.startswith(">"):
                    operator = ">"
                    value_str = condition[1:]
                elif condition.startswith("<"):
                    operator = "<"
                    value_str = condition[1:]
                else:
                    print(f"[EvaluatorAgent] Warning: Invalid condition '{condition}' for '{col}'. Use >=, <=, >, <.")
                    continue

                val = float(value_str)
                
                # Apply filter (Update Qualified=False if condition VIOLATED)
                if operator == ">=":
                    combined_df.loc[combined_df[col] < val, "Qualified"] = False
                elif operator == "<=":
                    combined_df.loc[combined_df[col] > val, "Qualified"] = False
                elif operator == ">":
                    combined_df.loc[combined_df[col] <= val, "Qualified"] = False
                elif operator == "<":
                    combined_df.loc[combined_df[col] >= val, "Qualified"] = False
                    
                applied_constraints.append(f"{col} {condition}")
                
            except Exception as e:
                print(f"[EvaluatorAgent] Error applying constraint {col} {condition}: {e}")
                
        # 5. Update State & Persist
        # Update in-memory all_molecules_df
        self.all_molecules_df = combined_df.copy()
        
        # Update current_molecules_df (subset)
        if "smiles" in combined_df.columns:
            status_map = combined_df.set_index("smiles")["Qualified"].to_dict()
            def _update_status(row):
                smi = row.get("smiles")
                if smi and smi in status_map:
                    return status_map[smi]
                return row.get("Qualified", False)
            self.current_molecules_df["Qualified"] = self.current_molecules_df.apply(_update_status, axis=1)
            
        # Write back to CSV to save the 'Qualified' status updates
        try:
            combined_df.to_csv(summary_path, index=False)
            print(f"[EvaluatorAgent] Updated 'Qualified' status in {summary_path}")
        except Exception as e:
            print(f"[EvaluatorAgent] Error saving updated CSV: {e}")
            
        qualified_count = len(combined_df[combined_df["Qualified"] == True])
        
        return f"Filtering applied on accumulated data. Constraints: {applied_constraints}. Total Qualified (Accumulated): {qualified_count}."

    # --- Helper Methods ---

    def _compute_mpo(self, df, dock_min=-12.0, dock_max=-4.0, sa_min=1.0, sa_max=10.0, weights=None):
        """
        Compute an MPO_Score for each row in the dataframe.
        MPO_Score = w_docking * Docking_Norm + w_qed * QED + w_sa * SA_Norm
        Docking_Norm maps Docking_Score from [dock_min..dock_max] -> [1..0]
        SA_Norm maps SA from [sa_min..sa_max] -> [1..0]
        Weights are read from environment variables if not supplied explicitly and are normalized to sum to 1.
        """
        # Read weights from env if not provided
        if weights is None:
            try:
                w_d = float(os.getenv("EVALUATOR_MPO_WEIGHT_DOCKING", "0.5"))
                w_q = float(os.getenv("EVALUATOR_MPO_WEIGHT_QED", "0.3"))
                w_s = float(os.getenv("EVALUATOR_MPO_WEIGHT_SA", "0.2"))
                total = w_d + w_q + w_s
                if total <= 0:
                    w_d, w_q, w_s = 0.5, 0.3, 0.2
                else:
                    w_d, w_q, w_s = w_d / total, w_q / total, w_s / total
                weights = {"docking": w_d, "qed": w_q, "sa": w_s}
            except Exception as e:
                print(f"[EvaluatorAgent] Error parsing MPO weights from env: {e}; using defaults.")
                weights = {"docking": 0.5, "qed": 0.3, "sa": 0.2}
        else:
            # Normalize provided weights
            try:
                total = sum(weights.values())
                if total > 0:
                    weights = {k: float(v) / total for k, v in weights.items()}
                else:
                    weights = {"docking": 0.5, "qed": 0.3, "sa": 0.2}
            except Exception:
                weights = {"docking": 0.5, "qed": 0.3, "sa": 0.2}

        df = df.copy()
        # Ensure numeric columns exist
        df["Docking_Score"] = pd.to_numeric(df.get("Docking_Score", pd.Series([np.nan] * len(df))), errors="coerce")
        df["QED"] = pd.to_numeric(df.get("QED", pd.Series([0.0] * len(df))), errors="coerce").fillna(0.0)
        df["SA"] = pd.to_numeric(df.get("SA", pd.Series([sa_max] * len(df))), errors="coerce").fillna(sa_max)

        # Docking normalization: dock_min -> 1.0, dock_max -> 0.0
        def _norm_dock(x):
            if pd.isna(x):
                return 0.0
            # clip to range
            x_clipped = max(min(x, dock_max), dock_min)
            # (x - dock_max) / (dock_min - dock_max) maps dock_max->0, dock_min->1
            return max(0.0, min(1.0, (x_clipped - dock_max) / (dock_min - dock_max)))

        df["Docking_Norm"] = df["Docking_Score"].apply(_norm_dock)
        df["SA_Norm"] = df["SA"].apply(lambda x: max(0.0, min(1.0, (sa_max - x) / (sa_max - sa_min))))

        df["MPO_Score"] = (
            df["Docking_Norm"] * weights["docking"] +
            df["QED"] * weights["qed"] +
            df["SA_Norm"] * weights["sa"]
        )
        print(f"[EvaluatorAgent] Using MPO weights: {weights}")
        return df

    def _get_mol(self, row):
        if "mol_obj" in row and row["mol_obj"]:
            return row["mol_obj"]
        if "smiles" in row:
            return Chem.MolFromSmiles(row["smiles"])
        return None

    def _prepare_receptor(self, pdb_path, output_path):
        # Use openbabel or MGLTools
        # Simple openbabel conversion
        cmd = ["obabel", pdb_path, "-O", output_path, "-xr"] # -xr to keep rigid
        subprocess.run(cmd, check=False)

    def _prepare_ligand(self, mol, output_path, use_openbabel=False):
        # 1. Add Hydrogens
        mol = Chem.AddHs(mol)
        # 2. Generate 3D Conformer
        # Use ETKDGv3 for better conformer generation
        params = AllChem.ETKDGv3()
        params.useRandomCoords = True
        res = AllChem.EmbedMolecule(mol, params)
        
        # If embedding fails, try random coordinates
        if res == -1:
             res = AllChem.EmbedMolecule(mol, useRandomCoords=True)
             
        # If still fails, return (will be handled by caller)
        if res == -1:
             print(f"[EvaluatorAgent] Failed to embed molecule: {Chem.MolToSmiles(mol)}")
             return

        # Helper for OpenBabel conversion
        def run_openbabel():
            try:
                temp_sdf = output_path.replace(".pdbqt", ".sdf")
                writer = Chem.SDWriter(temp_sdf)
                writer.write(mol)
                writer.close()
                subprocess.run(["obabel", temp_sdf, "-O", output_path], check=False)
                if os.path.exists(temp_sdf):
                    os.remove(temp_sdf)
            except Exception as e:
                print(f"[EvaluatorAgent] OpenBabel preparation failed: {e}")

        if use_openbabel:
            run_openbabel()
            return

        # 3. Convert to PDBQT using Meeko
        try:
            from meeko import MoleculePreparation
            from meeko import PDBQTMolecule
            from meeko import PDBQTWriterLegacy
            
            preparator = MoleculePreparation()
            # Meeko v0.5+ returns a list of setups
            mol_setups = preparator.prepare(mol)
            if mol_setups:
                pdbqt_string = PDBQTWriterLegacy.write_string(mol_setups[0])
                if isinstance(pdbqt_string, tuple):
                    pdbqt_string = pdbqt_string[0]
            else:
                # Fallback for older versions or empty result
                pdbqt_string = preparator.write_pdbqt_string()
            
            with open(output_path, "w") as f:
                f.write(pdbqt_string)
        except Exception as e:
            print(f"[EvaluatorAgent] Meeko preparation failed: {e}. Falling back to OpenBabel.")
            run_openbabel()

    def _run_single_docking(self, tool, receptor, ligand, center):
        # Load environment variables for configuration (Moved outside or cached if possible, but safe here)
        # Note: load_dotenv() in every thread is IO intensive but safe.
        # We'll rely on env vars being set in parent or default.
        # Use DOCKING_CPU_CORES for both Vina and Gnina
        cpu_cores = os.getenv("DOCKING_CPU_CORES", "1")

        out_file = ligand.replace(".pdbqt", "_out.pdbqt")
        log_file = ligand.replace(".pdbqt", ".log")
        
        try:
            cmd = []
            is_gnina = "gnina" in tool
            
            if is_gnina:
                # Gnina command construction
                cmd = [
                    tool,
                    "--receptor", receptor,
                    "--ligand", ligand,
                    "--center_x", str(center[0]),
                    "--center_y", str(center[1]),
                    "--center_z", str(center[2]),
                    "--size_x", "20", "--size_y", "20", "--size_z", "20",
                    "--out", out_file,
                    "--log", log_file,
                    "--cpu", str(cpu_cores), 
                    "--exhaustiveness", "8",
                    "--no_gpu" # Force CPU mode for gnina as requested
                ]
            else:
                # Vina command construction
                cmd = [
                    tool,
                    "--receptor", receptor,
                    "--ligand", ligand,
                    "--center_x", str(center[0]),
                    "--center_y", str(center[1]),
                    "--center_z", str(center[2]),
                    "--size_x", "20", "--size_y", "20", "--size_z", "20",
                    "--out", out_file,
                    "--cpu", str(cpu_cores), 
                    "--exhaustiveness", "8"
                ]
                # Vina writes log to stdout/stderr usually, but can use > redirect if run via shell
                # Or some versions support --log. Official Vina supports --log.
                # cmd.extend(["--log", log_file]) # Removed: installed vina version doesn't support --log
            
            # Handle LD_LIBRARY_PATH for gnina (local or system)
            env = os.environ.copy()
            if is_gnina:
                # Add conda lib path to ensure gnina finds necessary libraries (even for CPU mode)
                conda_prefix = os.environ.get("CONDA_PREFIX")
                lib_paths = []
                
                if conda_prefix:
                    lib_paths.append(os.path.join(conda_prefix, "lib"))

                fallback_lib = os.path.join(sys.prefix, "lib")
                if os.path.exists(fallback_lib) and fallback_lib not in lib_paths:
                    lib_paths.append(fallback_lib)

                current_ld = env.get("LD_LIBRARY_PATH", "")
                new_ld = ":".join(lib_paths + [current_ld])
                env["LD_LIBRARY_PATH"] = new_ld
                
                # Use LD_PRELOAD if needed (kept from previous fixes, though CPU mode might not need it as much)
                if conda_prefix:
                    preload_libs = [
                        os.path.join(conda_prefix, "lib", "libcudart.so.12"),
                        os.path.join(conda_prefix, "lib", "libcublas.so.12"),
                        os.path.join(conda_prefix, "lib", "libcublasLt.so.12"),
                        os.path.join(conda_prefix, "lib", "libcufft.so.11"),
                        os.path.join(conda_prefix, "lib", "libcusparse.so.12"),
                        os.path.join(conda_prefix, "lib", "libcusolver.so.11"),
                        os.path.join(conda_prefix, "lib", "libnvToolsExt.so.1"),
                    ]
                    preload_libs = [lib for lib in preload_libs if os.path.exists(lib)]
                    if preload_libs:
                        env["LD_PRELOAD"] = ":".join(preload_libs)
                
                print(f"[EvaluatorAgent] Running gnina (CPU mode) with {cpu_cores} cores")
            else:
                print(f"[EvaluatorAgent] Running vina with {cpu_cores} cores")

            # Run docking command
            result = subprocess.run(cmd, check=True, capture_output=True, env=env)
            
            # For Vina, write stdout to log file (since we removed --log)
            if not is_gnina:
                 with open(log_file, "w") as f:
                     f.write(result.stdout.decode('utf-8'))
            
            # Parse score from log
            with open(log_file, "r") as f:
                for line in f:
                    if line.strip().startswith("1"):
                        parts = line.split()
                        if len(parts) >= 2:
                            return float(parts[1])
            print(f"[EvaluatorAgent] Warning: Could not parse score from {log_file}")
            return 0.0
        except subprocess.CalledProcessError as e:
            print(f"[EvaluatorAgent] Docking subprocess failed with return code {e.returncode}")
            # print(f"[EvaluatorAgent] Stdout: {e.stdout}") # Reduce verbosity
            print(f"[EvaluatorAgent] Stderr: {e.stderr.decode('utf-8') if e.stderr else 'No stderr'}")
            return 0.0
        except Exception as e:
            print(f"[EvaluatorAgent] Docking failed with exception: {e}")
            return 0.0

# LangGraph Node Wrapper
def evaluator_agent_node(state: AgentState) -> Dict[str, Any]:
    agent = EvaluatorAgent()
    return agent.run(state)
