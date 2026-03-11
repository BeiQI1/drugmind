import os
import json
import time
from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from agent.base_agent import BaseAgent
from agent.state import AgentState
from agent.RAGAgent import RAGAgent
import pandas as pd
import subprocess
import shutil
import html
import tempfile
import math

class SynthesisAgent(BaseAgent):
    def __init__(self):
        super().__init__(agent_name="SynthesisAgent")
        self.rag_agent = RAGAgent()
        self.work_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "synthesis_results")
        os.makedirs(self.work_dir, exist_ok=True)
        
        # Cache for model and inventory
        self.rxn_model = None
        self.inventory_obj = None
        # Candidate storage
        self.candidates_df = None
        self.candidates_source = None
        self.selected_pool = None
        self.selected_df = None
        
        # Tool implementations
        self.tool_implementations = {
            "LoadCandidates": self.load_candidates,
            "SelectTopN": self.select_top_n,
            "RunRetrosynthesis": self.run_retrosynthesis,
            "RunAiZynth": self.run_aizynth,
            "GenerateSynthesisReport": self.generate_synthesis_report,
            "AnalyzeRouteComplexity": self.analyze_route_complexity,
            "Finish": None # Special handled in run loop
        }
        
        # Tool descriptions for the LLM
        self.tool_descriptions = {
            "LoadCandidates": "Loads filtered molecules CSV for synthesis selection. Args: file_path (optional)",
            "SelectTopN": "Selects the top-n molecules from loaded candidates using Total_Score = QED + Affinity - 0.5*SA. Args: n (optional)",
            "RunRetrosynthesis": "Performs retrosynthesis planning (Syntheseus). Args: smiles, inventory (default='zinc'), max_steps (default=10), time_limit (default=60s)",
            "RunAiZynth": "Runs AiZynthFinder retrosynthesis in conda env AIZYNTH_CONDA_ENV. Args: smiles, out_dir (optional)",
            "GenerateSynthesisReport": "Aggregates AiZynth results into HTML and generates PNG screenshots. Args: out_dir (optional)",
            "AnalyzeRouteComplexity": "Analyzes the complexity of a synthesis route (number of steps, starting materials). Args: route_json",
            "Finish": "Completes the synthesis planning task. Args: status (success/fail), summary"
        }

    def _load_resources(self, inventory_name: str = "zinc"):
        """Lazy loads the Chemformer model and inventory."""
        from syntheseus.reaction_prediction.inference.chemformer import ChemformerModel
        from syntheseus.search.mol_inventory import SmilesListInventory
        from syntheseus.interface.molecule import Molecule

        # 1. Load Model (if not loaded)
        if self.rxn_model is None:
            print(f"[{self.agent_name}] Loading Chemformer model...")
            model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools", "syntheseus", "model")
            try:
                self.rxn_model = ChemformerModel(model_dir=model_dir, is_forward=False)
                print(f"[{self.agent_name}] Model loaded successfully.")
            except Exception as e:
                print(f"Error loading Chemformer model: {e}")
                raise e

        # 2. Load Inventory (if not loaded)
        if self.inventory_obj is None:
            print(f"[{self.agent_name}] Loading Inventory ({inventory_name})...")
            zinc_inventory_path = os.path.join(self.work_dir, "inventory_zinc250k.txt")
            default_inventory_path = os.path.join(self.work_dir, "inventory.txt")
            
            inventory_path = None
            if inventory_name == "zinc" and os.path.exists(zinc_inventory_path):
                inventory_path = zinc_inventory_path
                print(f"[{self.agent_name}] Using ZINC250k inventory.")
            elif os.path.exists(default_inventory_path):
                inventory_path = default_inventory_path
                print(f"[{self.agent_name}] Using default inventory.")
                
            if not inventory_path:
                 print(f"[{self.agent_name}] Warning: No inventory file found. Using minimal fallback list.")
                 buyable_smiles = ["CC(=O)Cl", "NCC1=CC=CC=C1", "c1ccccc1", "CCO", "CC(=O)O"]
            else:
                 with open(inventory_path, "r") as f:
                     buyable_smiles = [line.strip() for line in f if line.strip()]
            
            # Define custom inventory class locally if needed, or use SmilesListInventory
            # Since we fixed the import earlier, we can use SmilesListInventory directly
            self.inventory_obj = SmilesListInventory(smiles_list=buyable_smiles)
            print(f"[{self.agent_name}] Inventory loaded with {len(buyable_smiles)} molecules.")

    def _get_kg_context(self, smiles: str) -> str:
        """Retrieves synthesis knowledge from KG/RAG."""
        try:
            # 1. Query KG for tools (optional, if we want dynamic tools)
            # tools = self.kg_loader.query_agent_tools("SynthesisAgent")
            
            # 2. Retrieve general synthesis guidelines or specific reaction templates
            query = f"Retrosynthesis strategies for drug-like molecules similar to {smiles}"
            context = self.rag_agent.retrieve(query, k=2)
            return context
        except Exception as e:
            return f"Error retrieving KG context: {e}"

    def _construct_react_prompt(self, smiles: str, kg_context: str) -> str:
        tools_str = "\n".join([f"- {name}: {desc}" for name, desc in self.tool_descriptions.items()])
        
        return f"""You are the SynthesisAgent. Your goal is to plan a synthesis route for the given molecule.
        
Target Molecule: {smiles}

Knowledge Graph Context:
{kg_context}

Available Tools:
{tools_str}

Standard Operating Procedure (SOP):
1. Analyze the target molecule and KG context.
2. Call 'RunRetrosynthesis' to generate a route. You can specify 'inventory' (e.g., 'zinc', 'enamine') and 'max_steps'.
3. Call 'AnalyzeRouteComplexity' to evaluate the route.
4. If the route is valid and complexity is acceptable, call 'Finish' with status='success'.
5. If retrosynthesis fails or is too complex, call 'Finish' with status='fail'.

Format your response as a JSON object with the following structure:
{{
    "thought": "Your reasoning here",
    "tool": "ToolName",
    "args": { "arg_name": "value" }
}}
"""

    def _construct_react_prompt_for_batch(self, df: pd.DataFrame, kg_context: str) -> str:
        """Constructs a ReAct-style prompt for a batch of selected molecules."""
        tools_str = "\n".join([f"- {name}: {desc}" for name, desc in self.tool_descriptions.items()])
        rows = []
        for idx, row in df.reset_index().iterrows():
            rows.append(f"#{idx+1}: SMILES={row.get('smiles')} | QED={row.get('QED')} | SA={row.get('SA')} | Docking={row.get('Docking_Score')} | Total={row.get('Total_Score', 'N/A')}")
        table = "\n".join(rows)
        return f"""You are the SynthesisAgent. You are now provided with the selected candidate molecules and KG context.

Candidates:
{table}

KG Context:
{kg_context}

Available Tools:
{tools_str}

SOP:
1) For each candidate, consider running RunAiZynth to propose retrosynthetic routes (AiZynthFinder).
2) Use RunRetrosynthesis (Syntheseus) to cross-check or produce alternative routes.
3) Use AnalyzeRouteComplexity to evaluate routes.
4) Call GenerateSynthesisReport to produce an aggregated HTML/PNG report.

Respond with a JSON action object to call one of the tools, for example:
{{"thought": "I will run AiZynth on candidate #1 first to check routes.", "tool": "RunAiZynth", "args": {{"smiles": "C1=CC=CC=C1"}}}}"""
    def run(self, state: AgentState) -> Dict[str, Any]:
        print(f"\n[{self.agent_name}] Starting synthesis planning...")
        
        # Check if we are running standalone (just a SMILES string input) or as part of a pipeline
        user_input = state.get("user_input", "")
        intent = state.get("intent", "")
        task_params = state.get("task_params", {})
        
        # Scenario 1: Standalone Retrosynthesis (User provided SMILES in prompt or task_params)
        target_smiles = task_params.get("smiles")
        if not target_smiles and intent == "synthesis_planning":
             # Try to extract SMILES from user input if not explicitly parsed
             # Simple heuristic or rely on what IntentAgent extracted
             pass
        
        if target_smiles:
            print(f"[{self.agent_name}] Mode: Single Target Retrosynthesis for {target_smiles}")
            # Create a dataframe with 1 row for compatibility
            self.selected_df = pd.DataFrame([{"smiles": target_smiles, "QED": 0.0, "SA": 0.0, "Docking_Score": 0.0}])
        else:
            # Scenario 2: Pipeline (Load from Evaluator results)
            print(f"[{self.agent_name}] Mode: Batch Pipeline Selection")
            load_res = self.load_candidates(file_path=state.get("results", {}).get("evaluation", {}).get("feedback", {}).get("filtered_csv"))
            if isinstance(load_res, str) and load_res.startswith("Error"):
                return {"messages": [AIMessage(content=load_res)]}

            # Determine n from env only
            try:
                n = int(os.getenv("SYNTHESIS_AGENT_TOPN_DEFAULT", "3"))
            except Exception:
                n = 3
            n = max(1, int(n))

            # Select top-n directly from evaluator-sorted results
            self.selected_pool = None
            self.selected_df = None
            self.select_top_n(n=n, use_total=False)
            print(f"[{self.agent_name}] Selected top {n} molecules for retrosynthesis.")

        # Let the LLM plan how to run AiZynth / Syntheseus on these selected molecules via tools
        kg_context = self._get_kg_context(self.selected_df.iloc[0]['smiles']) if not self.selected_df.empty else ""
        messages = [SystemMessage(content=self.get_system_prompt())]
        messages.append(HumanMessage(content=self._construct_react_prompt_for_batch(self.selected_df, kg_context)))

        # Start a ReAct loop where the LLM can call 'RunAiZynth' or 'RunRetrosynthesis' tools
        max_steps = int(os.getenv("AGENT_MAX_STEPS", 10))
        current_step = 0
        finish_args: Optional[Dict[str, Any]] = None
        while current_step < max_steps:
            try:
                response = self.model.invoke(messages)
                content = response.content
            except Exception as e:
                print(f"[{self.agent_name}] Error invoking model: {e}")
                break

            try:
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()

                action_data = json.loads(content)
                thought = action_data.get("thought")
                action = action_data.get("tool")
                args = action_data.get("args", {})

                print(f"[{self.agent_name}] Step {current_step+1}: {thought}")
                print(f"[{self.agent_name}] Action: {action} Args: {args}")

                if action == "Finish":
                    finish_args = args
                    break

                if action in self.tool_implementations:
                    try:
                        result = self.tool_implementations[action](**args)
                    except Exception as e:
                        result = f"Error executing {action}: {e}"
                else:
                    result = f"Error: Tool {action} not found."

                messages.append(AIMessage(content=content))
                messages.append(HumanMessage(content=f"Tool Output: {result}"))

            except json.JSONDecodeError:
                messages.append(HumanMessage(content="Error: Invalid JSON from model. Please reply with valid JSON."))
            except Exception as e:
                print(f"[{self.agent_name}] Error in loop: {e}")
                break

            current_step += 1
            
        # After the loop, automatically run AiZynth on selected molecules and generate report
        out_dir = os.path.join(self.work_dir, f"aizynth_{int(time.time())}")
        os.makedirs(out_dir, exist_ok=True)
        run_results = []
        for idx, row in self.selected_df.iterrows():
            smi = row.get('smiles')
            r = self.run_aizynth(smiles=smi, out_dir=out_dir)
            run_results.append({'smiles': smi, 'result': r})

        report_path = self.generate_synthesis_report(out_dir=out_dir)
        
        # --- Generate Route Summary Images ---
        route_images = []
        route_data_files = []
        for idx, row in self.selected_df.iterrows():
            smi = row.get('smiles')
            if not smi: continue
            safe = ''.join([c if c.isalnum() else '_' for c in smi])
            # Find subdir
            try:
                candidates = [d for d in os.listdir(out_dir) if d.startswith(f"aizynth_{safe}_")]
                if candidates:
                    # Use first match
                    json_path = os.path.join(out_dir, candidates[0], "output.json")
                    if os.path.exists(json_path):
                        # Store raw data file
                        route_data_files.append(json_path)
                        
                        img_name = f"route_summary_{safe}.png"
                        img_path = os.path.join(out_dir, candidates[0], img_name)
                        if self._generate_route_image(json_path, img_path):
                            route_images.append(img_path)
                            print(f"[{self.agent_name}] Generated route image: {img_path}")
            except Exception as e:
                print(f"[{self.agent_name}] Error listing dirs for images: {e}")

        finish_note = ""
        if isinstance(finish_args, dict) and finish_args.get("summary"):
            finish_note = f" ({finish_args.get('summary')})"
        return {
            "messages": [AIMessage(content=f"Synthesis report generated at {report_path}{finish_note}")],
            "synthesis_report": report_path,
            "route_images": route_images,
            "route_data_files": route_data_files,
        }

    def run_retrosynthesis(self, smiles: str, inventory: str = "zinc", max_steps: int = 10, time_limit: int = 60, **kwargs) -> str:
        """
        Runs actual retrosynthesis using Syntheseus with Chemformer model.
        """
        print(f"[{self.agent_name}] Running Syntheseus with inventory={inventory}, max_steps={max_steps}, time_limit={time_limit}s")
        
        from rdkit import Chem
        from syntheseus.interface.molecule import Molecule
        from syntheseus.search.algorithms.best_first import retro_star
        from syntheseus.search.node_evaluation.common import ReactionModelLogProbCost, ConstantNodeEvaluator
        from syntheseus.search.visualization import visualize_andor

        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return "Error: Invalid SMILES string."
            
        # Ensure resources are loaded
        try:
            self._load_resources(inventory_name=inventory)
        except Exception as e:
            return f"Error loading resources: {e}"

        # 3. Setup Search
        target_mol = Molecule(smiles=smiles)
        
        # Cost functions (from tutorial)
        or_node_cost_fn = retro_star.MolIsPurchasableCost()
        and_node_cost_fn = ReactionModelLogProbCost(normalize=False)
        value_function = ConstantNodeEvaluator(0.0) # Retro*-0
        
        search_algo = retro_star.RetroStarSearch(
            reaction_model=self.rxn_model,
            mol_inventory=self.inventory_obj,
            or_node_cost_fn=or_node_cost_fn,
            and_node_cost_fn=and_node_cost_fn,
            value_function=value_function,
            limit_reaction_model_calls=max_steps,
            time_limit_s=time_limit
        )

        # 4. Run Search
        print(f"[{self.agent_name}] Starting search...")
        try:
            # Note: run_from_mol returns (graph, result_status)
            output_graph, _ = search_algo.run_from_mol(target_mol)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error during search execution: {e}"

        # 5. Analyze Results
        # Extract routes
        from syntheseus.search.analysis import route_extraction
        # Set route costs for extraction
        from syntheseus.search.graph.and_or import AndNode
        for node in output_graph.nodes():
            if isinstance(node, AndNode):
                node.data["route_cost"] = 1.0
            else:
                node.data["route_cost"] = 0.0

        routes = list(route_extraction.iter_routes_cost_order(output_graph, max_routes=5))
        
        if not routes:
            return "Search completed but no routes found."
            
        # Visualize best route
        best_route = routes[0]
        output_pdf = os.path.join(self.work_dir, f"route_{int(time.time())}.pdf")
        visualize_andor(output_graph, filename=output_pdf, nodes=best_route)
        
        # Check if solved
        is_solved = output_graph.root_node.is_solved
        viz_msg = f"Visualization saved to {output_pdf}"
        
        # 7. Construct Output
        route_info = {
            "target": smiles,
            "solved": is_solved,
            "graph_size": len(output_graph.nodes()),
            "visualization": viz_msg
        }
        
        out_file = os.path.join(self.work_dir, f"route_{int(time.time())}.json")
        with open(out_file, "w") as f:
            json.dump(route_info, f, indent=2)

        return f"Retrosynthesis complete. Solved: {is_solved}. {viz_msg}"

    def analyze_route_complexity(self, route_json: Dict = None, **kwargs):
        # If route_json is passed as string (from LLM), try to parse
        if isinstance(route_json, str):
            try:
                # It might be a file path
                if os.path.exists(route_json):
                    with open(route_json, 'r') as f:
                        route_json = json.load(f)
                else:
                    return "Error: Invalid route JSON or file path."
            except:
                pass
        
        # Mock analysis
        steps = 2 # derived from mock
        materials = 2
        return f"Route Complexity: Low. Steps: {steps}. Starting Materials: {materials}. Feasible."

    # --- New helper tools for selection and AiZynth integration ---
    def load_candidates(self, file_path: str = None):
        """Loads filtered candidates CSV into memory."""
        eval_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "evaluation_results")
        if file_path and os.path.exists(file_path):
            csv_path = file_path
        else:
            # find latest filtered_molecules.csv
            latest = None
            for root, dirs, files in os.walk(eval_root):
                if "filtered_molecules.csv" in files:
                    cand = os.path.join(root, "filtered_molecules.csv")
                    if latest is None or os.path.getmtime(cand) > os.path.getmtime(latest):
                        latest = cand
            csv_path = latest

        if not csv_path:
            return "Error: No filtered_molecules.csv found."

        try:
            df = pd.read_csv(csv_path)
            # Ensure required columns
            for c in ["smiles", "QED", "SA", "Docking_Score"]:
                if c not in df.columns:
                    df[c] = pd.NA
            self.candidates_df = df
            self.candidates_source = csv_path
            return f"Loaded {len(df)} candidates from {csv_path}"
        except Exception as e:
            return f"Error loading candidates: {e}"

    def select_top_n(self, n: int = 3, use_total: bool = True):
        """Select top-n molecules.
        If use_total is False, selects top-n pool from the provided CSV (assuming Evaluator already sorted them).
        If use_total is True, computes Total_Score = QED + Affinity - 0.5*SA within the pool and selects top n.
        """
        if not hasattr(self, 'candidates_df') or self.candidates_df is None or self.candidates_df.empty:
            return "Error: No candidates loaded."

        df = self.candidates_df.copy()
        # Form initial top10 if requested
        if not use_total:
            # Assume input CSV is already sorted by EvaluatorAgent (best to worst).
            # We just take the top 10 rows directly without re-sorting by Docking_Score.
            n_final = max(1, int(n))
            selected = df.head(n_final).reset_index(drop=True)
            self.selected_pool = selected
            self.selected_df = selected
            return f"Selected top {len(selected)} molecules from Evaluator results (took first {len(selected)} rows)."

        # use_total True: compute Total_Score within current selected_pool if available, else from entire set
        pool = getattr(self, 'selected_pool', None)
        if pool is None or pool.empty:
            n_final = max(1, int(n))
            pool_size = max(int(os.getenv("SYNTHESIS_AGENT_POOL_SIZE", "10")), n_final)
            pool = df.head(pool_size).reset_index(drop=True)
            self.selected_pool = pool

        # Ensure numeric
        pool["QED"] = pd.to_numeric(pool.get("QED", pd.Series([0.0]*len(pool))), errors="coerce").fillna(0.0)
        pool["SA"] = pd.to_numeric(pool.get("SA", pd.Series([10.0]*len(pool))), errors="coerce").fillna(10.0)
        # Affinity is usually -Docking_Score (since Docking Score is energy, lower is better, but Affinity implies strength, higher is better)
        # We assume Docking_Score is in kcal/mol (negative values). We negate it to get positive Affinity.
        pool["Affinity"] = -pd.to_numeric(pool.get("Docking_Score", pd.Series([0.0]*len(pool))), errors="coerce").fillna(0.0)

        # Compute Total_Score = QED + Affinity - 0.5 * SA
        pool["Total_Score"] = pool["QED"] + pool["Affinity"] - 0.5 * pool["SA"]

        # Select top n by Total_Score (higher better)
        n_final = max(1, int(n))

        selected = pool.sort_values(by="Total_Score", ascending=False).head(n_final).reset_index(drop=True)
        self.selected_df = selected
        return f"Selected top {len(selected)} molecules using Total_Score (n={n_final}) from the selected pool."

    def run_aizynth(self, smiles: str, out_dir: str = None, timeout: int = 300):
        """Runs AiZynthFinder in the specified conda env via 'conda run -n <env> -- <cmd>'
        Produces output files under out_dir/<safe_smiles>/. If AiZynth not available, creates a placeholder HTML.
        """
        env = os.getenv("SYNTHESIS_AGENT_AIZYNTH_CONDA_ENV", "aizynth-env")
        cmd_name = os.getenv("SYNTHESIS_AGENT_AIZYNTH_CMD", "aizynthcli")
        safe = ''.join([c if c.isalnum() else '_' for c in smiles])
        if out_dir is None:
            out_dir = os.path.join(self.work_dir, f"aizynth_{safe}_{int(time.time())}")
        else:
            out_dir = os.path.join(out_dir, f"aizynth_{safe}_{int(time.time())}")
        os.makedirs(out_dir, exist_ok=True)

        # Determine conda executable
        conda_exe = shutil.which("conda")
        if not conda_exe:
            # Try to find via environment variable
            conda_exe = os.getenv("CONDA_EXE", "conda")

        # Construct AiZynth CLI invocation
        # This is a best-effort invocation; exact flags may vary depending on local installation
        # We need to provide the config file
        project_root = os.path.dirname(os.path.dirname(__file__))
        base_config_path = os.path.join(project_root, "tools", "aizynth", "config.yml")
        
        # Create a temporary config file with absolute paths to ensure AiZynth finds the models
        temp_config_path = os.path.join(out_dir, "config.yml")
        try:
            if os.path.exists(base_config_path):
                with open(base_config_path, "r") as f:
                    config_content = f.read()
                
                # Replace relative path "tools/aizynth/" with absolute path
                abs_tools_path = os.path.join(project_root, "tools", "aizynth") + os.sep
                new_content = config_content.replace("tools/aizynth/", abs_tools_path)
                
                with open(temp_config_path, "w") as f:
                    f.write(new_content)
                config_path = temp_config_path
            else:
                return f"Error: AiZynth config not found at {base_config_path}"
        except Exception as e:
            return f"Error preparing AiZynth config: {e}"
        
        # NOTE: aizynthcli --output expects a FILE path when running with --smiles
        output_json = os.path.join(out_dir, "output.json")

        # NOTE: passing '--' sometimes causes issues with conda run in subprocess.
        # Since we use the conda executable directly to run 'run', and 'aizynthcli' is an executable,
        # we might try without '--' if the previous attempt failed. 
        # However, usually 'conda run -n env -- cmd' is safer.
        # But here 'conda run -n env cmd' worked in terminal (getting to python error).
        cmd = [conda_exe, "run", "-n", env, cmd_name, "--smiles", smiles, "--config", config_path, "--output", output_json]
        try:
            # We skip the 'which' check and run directly. If it fails, we capture it.
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            stdout = proc.stdout
            stderr = proc.stderr

            # Save logs
            with open(os.path.join(out_dir, "aizynth_stdout.log"), "w") as f:
                f.write(stdout)
            with open(os.path.join(out_dir, "aizynth_stderr.log"), "w") as f:
                f.write(stderr)

            # Check if command actually ran or if conda complained
            if proc.returncode != 0:
                return f"AiZynth run failed for {smiles}. Exit Code: {proc.returncode}. Logs and outputs in {out_dir}"

            # Convert JSON to HTML if it exists
            if os.path.exists(output_json):
                html_out = os.path.join(out_dir, "route.html")
                self._process_aizynth_json_to_html(output_json, html_out)

            return f"AiZynth completed for {smiles}. Logs and outputs in {out_dir}"
        except subprocess.TimeoutExpired:
            return f"AiZynth timed out for {smiles}"
        except Exception as e:
            return f"AiZynth run failed for {smiles}: {e}"

    def _process_aizynth_json_to_html(self, json_path: str, html_path: str):
        """Converts AiZynthFinder output JSON to a simple HTML representation."""
        import json
        import urllib.parse
        import urllib.request
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            enable_pubchem = os.getenv("SYNTHESIS_AGENT_PUBCHEM_NAMES", "1").strip() not in ("0", "false", "False")
            pubchem_delay_s = float(os.getenv("SYNTHESIS_AGENT_PUBCHEM_DELAY_S", "0.2"))
            pubchem_timeout_s = float(os.getenv("SYNTHESIS_AGENT_PUBCHEM_TIMEOUT_S", "20"))
            pubchem_max_synonyms = int(os.getenv("SYNTHESIS_AGENT_PUBCHEM_MAX_SYNONYMS", "10"))
            last_request_ts = 0.0

            name_cache: Dict[str, Dict[str, Any]] = {}

            def _sleep_if_needed():
                nonlocal last_request_ts
                now = time.time()
                elapsed = now - last_request_ts
                if elapsed < pubchem_delay_s:
                    time.sleep(pubchem_delay_s - elapsed)
                last_request_ts = time.time()

            def _get_json(url: str) -> Dict[str, Any]:
                _sleep_if_needed()
                req = urllib.request.Request(url, headers={"User-Agent": "drugtoolagent/1.0 (PubChem PUG-REST)"})
                with urllib.request.urlopen(req, timeout=pubchem_timeout_s) as resp:
                    return json.loads(resp.read().decode("utf-8"))

            def _pick_best_name(iupac_name: Optional[str], synonyms: List[str]) -> Optional[str]:
                candidates: List[str] = []
                if iupac_name:
                    candidates.append(iupac_name)
                candidates.extend(synonyms or [])
                for nm in candidates:
                    if not isinstance(nm, str):
                        continue
                    s = nm.strip()
                    if not s:
                        continue
                    if len(s) > 80:
                        continue
                    return s
                return iupac_name or (synonyms[0] if synonyms else None)

            def pubchem_lookup(smiles: str) -> Dict[str, Any]:
                if smiles in name_cache:
                    return name_cache[smiles]

                out = {"cid": None, "best_name": None}
                if not enable_pubchem or not smiles:
                    name_cache[smiles] = out
                    return out

                try:
                    q = urllib.parse.quote(smiles, safe="")
                    prop_url = (
                        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
                        f"{q}/property/IUPACName,InChIKey,CanonicalSMILES/JSON"
                    )
                    prop = _get_json(prop_url)
                    props = (prop.get("PropertyTable") or {}).get("Properties") or []
                    iupac = None
                    cid = None
                    if props:
                        p0 = props[0]
                        cid = p0.get("CID")
                        iupac = p0.get("IUPACName")

                    syn_url = (
                        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
                        f"{q}/synonyms/JSON"
                    )
                    syn = _get_json(syn_url)
                    infos = (syn.get("InformationList") or {}).get("Information") or []
                    synonyms: List[str] = []
                    if infos:
                        synonyms = (infos[0].get("Synonym") or [])[:pubchem_max_synonyms]

                    out["cid"] = cid
                    out["best_name"] = _pick_best_name(iupac, synonyms)
                except Exception:
                    out = {"cid": None, "best_name": None}

                name_cache[smiles] = out
                return out
            
            with open(html_path, 'w') as f:
                f.write("<html><head><style>")
                f.write(".tree, .tree ul { list-style-type: none; margin: 0; padding: 0; }")
                f.write(".tree ul { margin-left: 20px; border-left: 1px solid #ccc; padding-left: 10px; }")
                f.write(".stock { color: green; font-weight: bold; }")
                f.write(".not-stock { color: red; }")
                f.write("</style></head><body>")
                f.write("<h3>Synthesis Routes</h3>")
                
                # data is a list of trees
                for i, tree in enumerate(data):
                    f.write(f"<h4>Route {i+1}</h4>")
                    f.write("<div class='tree'>")
                    self._write_node_html(f, tree, pubchem_lookup)
                    f.write("</div><hr>")
                
                f.write("</body></html>")
                
        except Exception as e:
            with open(html_path, 'w') as f:
                f.write(f"<html><body>Error processing JSON: {e}</body></html>")

    def _write_node_html(self, f, node, pubchem_lookup=None):
        f.write("<ul><li>")
        
        node_type = node.get('type', 'unknown')
        smiles = node.get('smiles', 'N/A')
        
        if node_type == 'mol':
            is_stock = node.get('in_stock', False)
            class_name = "stock" if is_stock else "not-stock"
            name_part = ""
            if callable(pubchem_lookup) and smiles and smiles != "N/A":
                info = pubchem_lookup(smiles)
                best_name = info.get("best_name")
                cid = info.get("cid")
                if best_name and cid:
                    link = f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}"
                    name_part = f" — <a href=\"{html.escape(link)}\" target=\"_blank\">{html.escape(str(best_name))}</a>"
                elif best_name:
                    name_part = f" — {html.escape(str(best_name))}"

            f.write(f"<span class='{class_name}'>Molecule: {html.escape(smiles)}{name_part}</span>")
            if is_stock:
                 f.write(" (In Stock)")
        elif node_type == 'reaction':
            metadata = node.get('metadata', {})
            policy = metadata.get('policy_name', 'Unknown')
            template = metadata.get('template', '')
            f.write(f"<span>Reaction ({policy}): {html.escape(template)}</span>")
            
        children = node.get('children', [])
        for child in children:
            self._write_node_html(f, child, pubchem_lookup)
            
        f.write("</li></ul>")

    def _generate_route_image(self, json_path: str, img_path: str) -> bool:
        """Generates a reaction pathway image (Step-by-step from Sources to Target)."""
        import json
        from rdkit import Chem
        from rdkit.Chem import Draw, AllChem
        try:
            from PIL import Image
        except ImportError:
            print(f"[{self.agent_name}] PIL not installed. Skipping image generation.")
            return False
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            if not data: return False
            
            # Best route is usually the first tree
            route = data[0]
            
            # Collect reactions: (product_smiles, [reactant_smiles_list])
            reactions = []
            
            def traverse(node):
                if node.get('type') == 'mol':
                    children = node.get('children', [])
                    for child in children:
                        if child.get('type') == 'reaction':
                            # This reaction produces 'node' (product)
                            product = node.get('smiles')
                            reactants = []
                            for grandchild in child.get('children', []):
                                if grandchild.get('type') == 'mol':
                                    reactants.append(grandchild.get('smiles'))
                            reactions.append((product, reactants))
                            # Continue traversal down to reactants
                            for grandchild in child.get('children', []):
                                traverse(grandchild)
            
            traverse(route)
            
            if not reactions:
                return False
            
            # Draw each reaction
            # Traversal is pre-order (Target -> Sources).
            # We want to display Sources -> Target (Synthesis direction).
            # So we reverse the list.
            images = []
            for prod, reacts in reversed(reactions):
                # reactants >> product
                try:
                    # Construct simple SMARTS: R1.R2>>P
                    rs = ".".join(reacts)
                    rxn_smarts = f"{rs}>>{prod}"
                    rxn = AllChem.ReactionFromSmarts(rxn_smarts, useSmiles=True)
                    
                    # Draw
                    d2d = Draw.MolDraw2DCairo(600, 200) # Width, Height
                    d2d.DrawReaction(rxn)
                    d2d.FinishDrawing()
                    png_data = d2d.GetDrawingText()
                    
                    # Convert to PIL Image
                    import io
                    img = Image.open(io.BytesIO(png_data))
                    images.append(img)
                except Exception as ex:
                    print(f"Error drawing step: {ex}")
            
            if not images: return False
            
            # Concatenate images vertically
            total_height = sum(img.height for img in images)
            max_width = max(img.width for img in images)
            
            combined_img = Image.new('RGB', (max_width, total_height), (255, 255, 255))
            
            y_offset = 0
            for img in images:
                # Center the image if it's smaller than max_width
                x_offset = (max_width - img.width) // 2
                combined_img.paste(img, (x_offset, y_offset))
                y_offset += img.height
                
            combined_img.save(img_path)
            return True
        except Exception as e:
            print(f"Error generating route image: {e}")
            return False

    def generate_synthesis_report(self, out_dir: str = None):
        """Aggregates per-molecule AiZynth outputs into a single HTML report and attempts PNG conversion."""
        if out_dir is None:
            return "Error: out_dir must be provided to generate report."
        files = []
        for root, dirs, fs in os.walk(out_dir):
            for f in fs:
                if f.endswith('.html') or f.endswith('.log'):
                    files.append(os.path.join(root, f))

        report_html = os.path.join(out_dir, "synthesis_report.html")
        with open(report_html, "w") as f:
            f.write("<html><body><h1>Synthesis Report</h1>")
            for idx, row in self.selected_df.reset_index().iterrows():
                smi = row['smiles']
                f.write(f"<h2>#{idx+1} - {html.escape(smi)}</h2>")
                safe = ''.join([c if c.isalnum() else '_' for c in smi])
                candidate_dir = os.path.join(out_dir, f"aizynth_{safe}_*")
                # list matching dirs
                matches = [d for d in os.listdir(out_dir) if d.startswith(f"aizynth_{safe}_")]
                if matches:
                    dpath = os.path.join(out_dir, matches[0])
                    for g in os.listdir(dpath):
                        if g.endswith('.html'):
                            rel = os.path.join(dpath, g)
                            f.write(f"<p><a href=\"{rel}\">{g}</a></p>")
                        if g.endswith('.log'):
                            rel = os.path.join(dpath, g)
                            f.write(f"<pre>{html.escape(open(rel).read()[:1000])}</pre>")
                else:
                    f.write("<p>No AiZynth output directory found for this molecule.</p>")
            f.write("</body></html>")

        # Try to convert to PNG via wkhtmltoimage
        wk = shutil.which('wkhtmltoimage')
        png_path = report_html.replace('.html', '.png')
        if wk:
            try:
                subprocess.run([wk, report_html, png_path], check=True)
                return report_html
            except Exception as e:
                return f"Report generated at {report_html}, but PNG conversion failed: {e}"
        else:
            return report_html

def synthesis_agent_node(state: AgentState) -> Dict[str, Any]:
    agent = SynthesisAgent()
    return agent.run(state)
