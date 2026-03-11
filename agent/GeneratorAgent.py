import os
import json
import subprocess
import concurrent.futures
import requests
import time
from typing import Dict, Any, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from agent.base_agent import BaseAgent
from agent.state import AgentState
from rdkit import Chem

class GeneratorAgent(BaseAgent):
    def __init__(self, model: Optional[BaseChatModel] = None):
        super().__init__(agent_name="GeneratorAgent", model=model)
        # Define working directory for generated molecules
        self.work_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "generated_molecules")
        os.makedirs(self.work_dir, exist_ok=True)
        
        # Service Configuration
        self.services = {
            "DiffSBDD": {
                "port": 3002,
                "script": os.path.join(os.path.dirname(os.path.dirname(__file__)), "services", "diffsbdd_api", "run.sh"),
                "health_endpoint": "http://localhost:3002/health",
                "generate_endpoint": "http://localhost:3002/generate"
            },
            "DecompDiff": {
                "port": 3003,
                "script": os.path.join(os.path.dirname(os.path.dirname(__file__)), "services", "decompdiff_api", "run.sh"),
                "health_endpoint": "http://localhost:3003/health",
                "generate_endpoint": "http://localhost:3003/generate"
            },
            "MiDi": {
                "port": 3004,
                "script": os.path.join(os.path.dirname(os.path.dirname(__file__)), "services", "midi_api", "run.sh"),
                "health_endpoint": "http://localhost:3004/health",
                "generate_endpoint": "http://localhost:3004/generate"
            },
            "GenMol": {
                "port": 3005,
                "script": os.path.join(os.path.dirname(os.path.dirname(__file__)), "services", "genmol_api", "run.sh"),
                "health_endpoint": "http://localhost:3005/health",
                "generate_endpoint": "http://localhost:3005/generate"
            },
            "DiffGui": {
                "port": 3006,
                "script": os.path.join(os.path.dirname(os.path.dirname(__file__)), "services", "diffgui_api", "run.sh"),
                "health_endpoint": "http://localhost:3006/health",
                "generate_endpoint": "http://localhost:3006/generate"
            }
        }

        # Define Tools for LLM
        self.tool_implementations = {
            "RunDiffSBDD": self.run_diffsbdd,
            "RunDiffGui": self.run_diffgui,
            "RunGenMol": self.run_genmol,
            "RunMiDi": self.run_midi,
            "RunDecompDiff": self.run_decompdiff,
            "Finish": None
        }

        self.tool_descriptions = {
            "RunDiffSBDD": "Generates molecules using DiffSBDD (Structure-based). Args: pdb_path (str), num_samples (int), mode (str: 'generate', 'optimize', 'inpaint'), ref_ligand (str, optional), fragments (str, optional).",
            "RunDiffGui": "Generates molecules using DiffGui (Property-guided). Args: pocket_pdb_path (str), num_mols (int), guidance_type (str: 'uncertainty', 'qed', 'sa', 'logp', 'tpsa', 'aff'), guidance_weight (float), guidance_params (dict, optional), ligand_sdf_path (str, optional), frag_sdf_path (str, optional).",
            "RunGenMol": "Generates molecules using GenMol (Language/SMILES-based). Args: task (str: 'denovo', 'linker_design', 'scaffold_decoration', 'scaffold_morphing'), num_samples (int), fragment_smiles (str, optional).",
            "RunMiDi": "Generates molecules using MiDi (Conditional/Unconditional). Args: pdb_path (str), num_samples (int), mode (str: 'conditional').",
            "RunDecompDiff": "Generates molecules using DecompDiff (Fragment-based). Args: pdb_path (str), num_samples (int), prior_mode (str: 'beta_prior', 'ref_prior', 'subpocket'), ref_ligand (str, optional).",
            "Finish": "Call this when all requested generation tasks are completed."
        }

    def run(self, state: AgentState) -> Dict[str, Any]:
        """
        Executes the generation models using a ReAct loop driven by the LLM.
        """
        task_params = state.get("task_params", {})
        existing_results = state.get("results", {})
        if not isinstance(existing_results, dict):
            existing_results = {}
        target_data = existing_results.get("target_preparation", {})
        
        # Context for LLM
        history = []
        results = {}
        max_steps = 10
        step = 0
        
        # Track used services for cleanup
        used_services = set()
        
        print(f"[GeneratorAgent] Starting ReAct loop for task: {task_params}")

        try:
            while step < max_steps:
                step += 1
                
                # A. Construct Prompt (SOP-Driven)
                prompt = self._construct_prompt(task_params, target_data, history)
                
                # B. Call LLM
                try:
                    messages = [SystemMessage(content=prompt)]
                    response = self.model.invoke(messages)
                    content = response.content.strip()
                except Exception as e:
                    print(f"[GeneratorAgent] LLM Invocation Failed: {e}")
                    time.sleep(5)
                    try:
                        response = self.model.invoke(messages)
                        content = response.content.strip()
                    except Exception as e2:
                        history.append({"step": step, "error": "LLM Error", "content": str(e2)})
                        break
                
                # C. Parse Action
                action = self._parse_action(content)
                
                if not action:
                    print(f"[GeneratorAgent] Failed to parse action from: {content}")
                    history.append({"step": step, "error": "Parse Error", "content": content})
                    continue
                    
                tool_name = action.get("tool")
                args = action.get("args", {})
                thought = action.get("thought", "")
                
                print(f"[GeneratorAgent] Step {step}")
                print(f"  Thought: {thought}")
                print(f"  Tool: {tool_name} Args: {args}")
                
                # D. Check Finish
                if tool_name == "Finish":
                    print("[GeneratorAgent] Agent decided to finish.")
                    break
                
                # E. Execute Tool
                try:
                    result = self._execute_tool(tool_name, args)

                    # Retry if generation returned 0 molecules (stochastic failure)
                    if isinstance(result, dict):
                        is_success = result.get("success", False)
                        has_mols = False
                        if "molecules" in result and result["molecules"]:
                            has_mols = True
                        elif "count" in result and result["count"] > 0:
                            has_mols = True
                        
                        if not (is_success or has_mols) and tool_name.startswith("Run"):
                            print(f"[GeneratorAgent] Warning: {tool_name} returned 0 molecules. Retrying with adjustment...")
                            time.sleep(2)
                            
                            # Adjust args for retry
                            retry_args = args.copy()
                            if tool_name == "RunDiffSBDD":
                                # Drastically increase samples for DiffSBDD as it has low success rate with MOAD checkpoint
                                retry_args["num_samples"] = max(retry_args.get("num_samples", 20) * 5, 100)
                                print(f"[GeneratorAgent] Increased DiffSBDD samples to {retry_args['num_samples']} for retry.")
                            
                            result = self._execute_tool(tool_name, retry_args)
                    
                    # Track service usage
                    if tool_name.startswith("Run") and tool_name.replace("Run", "") in self.services:
                        used_services.add(tool_name.replace("Run", ""))
                
                    # F. Update History & Results
                    history.append({
                        "step": step,
                        "thought": thought,
                        "tool": tool_name,
                        "args": args,
                        "result": str(result)
                    })
                    
                    # Store successful generation results
                    if isinstance(result, dict):
                        # Check for explicit success flag or non-empty molecule list
                        is_success = result.get("success", False)
                        has_mols = False
                        if "molecules" in result and result["molecules"]:
                            has_mols = True
                        elif "count" in result and result["count"] > 0:
                            has_mols = True
                        
                        if is_success or has_mols:
                            results[tool_name.replace("Run", "")] = result
                        else:
                            # Handle case where generation returned 0 molecules (e.g. filtering too strict or model failure)
                            print(f"[GeneratorAgent] Warning: {tool_name} returned 0 molecules or failed.")
                            # We might want to retry with relaxed parameters here, but for now just log it.
                except Exception as e:
                     print(f"[GeneratorAgent] Error executing tool {tool_name}: {e}")
                     import traceback
                     traceback.print_exc()
                     history.append({"step": step, "error": str(e), "tool": tool_name})



        finally:
            # Cleanup services to release memory
            if used_services:
                print(f"[GeneratorAgent] Cleaning up used services: {used_services}")
                req_list = [{"tool": s} for s in used_services]
                self._shutdown_services(req_list)
            
        existing_results["generation"] = results
        return {
            "current_agent": "GeneratorAgent",
            "results": existing_results,
            "task_params": task_params
        }

    def _format_history(self, history):
        history_str = ""
        for item in history:
            history_str += f"Step {item['step']}:\n"
            history_str += f"  Thought: {item.get('thought')}\n"
            history_str += f"  Action: {item.get('tool')} {item.get('args')}\n"
            history_str += f"  Result: {item.get('result')}\n\n"
        return history_str

    def _construct_prompt(self, task_params, target_data, history):
        sop_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sop", "GeneratorAgent_SOP.md")
        api_ref_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "services", "API_REFERENCE.md")
        
        try:
            with open(sop_path, 'r') as f:
                sop_content = f.read()
        except FileNotFoundError:
            sop_content = "No SOP file found. Proceed with standard tools."

        try:
            with open(api_ref_path, 'r') as f:
                api_ref_content = f.read()
        except FileNotFoundError:
            api_ref_content = "No API Reference found."
        
        prompt = f"""
{sop_content}

## Technical Reference (Model APIs)
{api_ref_content}

Current Execution State:
- Task Parameters: {json.dumps(task_params, indent=2)}
- Target Data: {json.dumps(target_data, indent=2)}
- Execution History:
{self._format_history(history)}

Instruction:
Based on the SOP above, decide the next step.
Return your response in JSON format:
{{
  "thought": "analysis of current state based on SOP",
  "tool": "ToolName",
  "args": {{ "arg": "value" }}
}}
If you have completed the SOP, use tool "Finish".
"""
        return prompt

    def _construct_react_prompt(self, task_params, target_data, history):
        tools_desc_str = json.dumps(self.tool_descriptions, indent=2)
        history_str = ""
        for item in history:
            history_str += f"Step {item['step']}:\n"
            history_str += f"  Thought: {item.get('thought')}\n"
            history_str += f"  Action: {item.get('tool')} {item.get('args')}\n"
            history_str += f"  Result: {item.get('result')}\n\n"

        task_params_for_prompt = dict(task_params) if isinstance(task_params, dict) else {"task_params": task_params}
        if "agent_logs" in task_params_for_prompt:
            task_params_for_prompt = {k: v for k, v in task_params_for_prompt.items() if k != "agent_logs"}
            
        prompt = f"""
You are the GeneratorAgent, an expert in AI-based drug design.
Your goal is to execute molecule generation tasks using the SPECIFIC models selected by the IntentAgent.

### Task Context:
- **User Intent**: {json.dumps(task_params_for_prompt, indent=2)}
- **Available Data (TargetAgent Output)**: {json.dumps(target_data, indent=2)}

### Available Tools:
{tools_desc_str}

### Instructions:
1. **Identify Required Models**: Look at `User Intent -> tools`. You MUST run ALL models listed there. You MUST NOT run models that are not listed there.
2. **Configure & Execute**: For each required model, determine the optimal parameters based on the `User Intent` details (e.g., mode, constraints).
   - If `tools` includes "DiffSBDD":
     - If intent is "optimization", call `RunDiffSBDD` with `mode='optimize'`.
     - If intent is "linker_design", call `RunDiffSBDD` with `mode='inpaint'`.
     - Otherwise, use `mode='generate'`.
   - If `tools` includes "DiffGui":
     - Map `constraints` (e.g., QED, SA) to `guidance_type` and `guidance_params`.
     - Always pass `ref_ligand_path` to `ligand_sdf_path` if available.
   - If `tools` includes "GenMol", "MiDi", or "DecompDiff", configure them similarly.
3. **Map Parameters**:
   - Use file paths from 'Available Data' (e.g., `cleaned_pdb`, `pocket_pdb_path`, `ref_ligand_path`).
   - If a required file path is missing in 'Available Data', check if an alternative exists (e.g., use `original_pdb` if `cleaned_pdb` is missing).
   - **CRITICAL**: You must pass the EXACT file path strings from 'Available Data' to the tools.
   - **DO NOT** attempt to call "TargetAgent" or any preparation tools. If data is missing, use the best available file path or fail gracefully.
4. **Execute**: Output the JSON action.
5. **Finish**: When ALL tools in the `tools` list have been executed, call 'Finish'.

### Execution History:
{history_str}

### Output Format:
```json
{{
  "thought": "The IntentAgent selected 'DiffSBDD'. The user wants to optimize the reference ligand. I have the ref_ligand_path.",
  "tool": "RunDiffSBDD",
  "args": {{ 
      "pdb_path": "/path/to/cleaned.pdb", 
      "num_samples": 20, 
      "mode": "optimize", 
      "ref_ligand": "/path/to/ligand.sdf" 
  }}
}}
```
"""
        return prompt

    def _parse_action(self, content):
        try:
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            else:
                json_str = content
            return json.loads(json_str)
        except:
            return None

    def _execute_tool(self, tool_name: str, args: Dict) -> Any:
        if tool_name not in self.tool_implementations:
            return f"Error: Tool {tool_name} not found."
        
        func = self.tool_implementations[tool_name]
        if func is None: return "Finished"
        
        try:
            return func(**args)
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"

    # --- Tool Implementations ---

    def run_diffsbdd(self, pdb_path: str, num_samples: int = 20, mode: str = "generate", ref_ligand: str = None, fragments: str = None):
        tool = "DiffSBDD"
        # Prioritize CrossDocked full-atom conditional checkpoint (Recommended for SBDD)
        ckpt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "DiffSBDD", "checkpoints", "crossdocked_fullatom_cond.ckpt")
        
        # Fallback to MOAD if CrossDocked is missing
        if not os.path.exists(ckpt_path):
             print(f"[GeneratorAgent] Warning: CrossDocked checkpoint not found at {ckpt_path}. Checking for MOAD...")
             moad_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "DiffSBDD", "checkpoints", "moad_fullatom_cond.ckpt")
             if os.path.exists(moad_path):
                 print(f"[GeneratorAgent] Falling back to MOAD checkpoint: {moad_path}")
                 ckpt_path = moad_path
             else:
                 print(f"[GeneratorAgent] Error: No valid DiffSBDD checkpoint found!")
        
        payload = {
            "checkpoint": ckpt_path,
            "pdbfile": pdb_path,
            "n_samples": num_samples,
            "sanitize": True
        }
        
        if mode == "inpaint" and fragments:
            payload["mode"] = "inpaint"
            payload["fix_atoms"] = fragments
            payload["ref_ligand"] = fragments
        elif mode == "optimize" and ref_ligand:
            payload["mode"] = "optimize"
            payload["ref_ligand"] = ref_ligand
        else:
            # De Novo logic
            if ref_ligand:
                payload["ref_ligand"] = ref_ligand
            else:
                # Try to find pocket info if not provided explicitly (DiffSBDD usually needs ref_ligand or pocket center)
                # Here we assume the LLM passed a valid PDB. If it's a full protein without ref_ligand, DiffSBDD might fail.
                # We can try to auto-detect pocket if we have the helper, but let's rely on inputs first.
                pass

        self._ensure_service_running(tool)
        return self._call_service({"tool": tool, "payload": payload})

    def run_diffgui(self, pocket_pdb_path: str, num_mols: int = 20, guidance_type: str = "uncertainty", guidance_weight: float = 1e-4, guidance_params: Dict = None, ligand_sdf_path: str = "None", frag_sdf_path: str = "None"):
        tool = "DiffGui"
        
        # Default guidance params if not provided
        if guidance_params is None:
            guidance_params = {
                "logp": 2.0, "tpsa": 100.0, "sa": 1.0, "qed": 0.8, "aff": 12.0, "gui_strength": 3.0
            }
            
        # Limit num_mols
        if num_mols > 20: num_mols = 20
        
        payload = {
            "pocket_pdb_path": pocket_pdb_path,
            "num_mols": num_mols,
            "batch_size": 8,
            "gen_mode": "frag_cond" if frag_sdf_path != "None" else "denovo",
            "ligand_sdf_path": ligand_sdf_path,
            "frag_sdf_path": frag_sdf_path,
            "guidance_type": guidance_type,
            "guidance_weight": guidance_weight,
            **guidance_params
        }
        
        self._ensure_service_running(tool)
        return self._call_service({"tool": tool, "payload": payload})

    def run_genmol(self, task: str, num_samples: int = 20, fragment_smiles: str = None, randomness: float = None):
        tool = "GenMol"
        payload = {
            "task": task,
            "num_samples": num_samples
        }
        if fragment_smiles:
            payload["fragment"] = fragment_smiles
        if randomness:
            payload["randomness"] = randomness
            
        self._ensure_service_running(tool)
        return self._call_service({"tool": tool, "payload": payload})

    def run_midi(self, pdb_path: str, num_samples: int = 20, mode: str = "conditional"):
        tool = "MiDi"
        payload = {
            "pdb_path": pdb_path,
            "num_samples": num_samples,
            "mode": mode
        }
        self._ensure_service_running(tool)
        return self._call_service({"tool": tool, "payload": payload})

    def run_decompdiff(self, pdb_path: str, num_samples: int = 20, prior_mode: str = "beta_prior", ref_ligand: str = None):
        tool = "DecompDiff"
        payload = {
            "pdb_path": pdb_path,
            "num_samples": num_samples,
            "prior_mode": prior_mode
        }
        self._ensure_service_running(tool)
        return self._call_service({"tool": tool, "payload": payload})

    # --- Service Management (Kept as is) ---
    def _ensure_service_running(self, tool: str, log_dir: str = None, timeout: int = 60):
        """Checks if service is running, if not starts it."""
        service_config = self.services[tool]
        health_url = service_config["health_endpoint"]
        
        if log_dir is None:
            log_dir = self.work_dir
        
        # Check if already running
        try:
            resp = requests.post(health_url, timeout=2)
            if resp.status_code == 200:
                return
        except requests.RequestException:
            pass
            
        # Start service
        print(f"[GeneratorAgent] Starting {tool} service...")
        script_path = service_config["script"]
        log_file = open(os.path.join(log_dir, f"{tool}_service.log"), "w")
        
        # Use preexec_fn to set process group, allowing us to kill the whole tree later
        subprocess.Popen(["bash", script_path], stdout=log_file, stderr=log_file, preexec_fn=os.setsid)
        
        # Wait for startup
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                resp = requests.post(health_url, timeout=2)
                if resp.status_code == 200:
                    print(f"[GeneratorAgent] {tool} service started successfully.")
                    return
            except requests.RequestException:
                time.sleep(2)
                
        raise RuntimeError(f"Failed to start {tool} service within {timeout} seconds.")

    def _shutdown_services(self, requests_list: List[Dict]):
        """Shuts down the services used in this run."""
        print("[GeneratorAgent] Shutting down services...")
        for req in requests_list:
            tool = req["tool"]
            port = self.services[tool]["port"]
            # Find process using the port and kill it
            # Using lsof or fuser is common, or pkill -f
            # Here we use a safer approach finding the PID listening on the port
            try:
                # Find PID listening on port
                # Use -sTCP:LISTEN to only get the server process, avoiding killing the client (self)
                cmd = f"lsof -t -i:{port} -sTCP:LISTEN"
                pid_output = subprocess.check_output(cmd, shell=True).decode().strip()
                if pid_output:
                    # Handle multiple PIDs (e.g. parent and child)
                    pids = pid_output.split('\n')
                    for pid in pids:
                        if pid.strip():
                            subprocess.run(f"kill -9 {pid}", shell=True)
                            print(f"[GeneratorAgent] Stopped {tool} service (PID {pid}).")
            except subprocess.CalledProcessError:
                pass # Process might already be gone
            except Exception as e:
                print(f"[GeneratorAgent] Warning: Failed to stop {tool}: {e}")

    def _call_service(self, request_item: Dict) -> Any:
        """Calls the service API."""
        tool = request_item["tool"]
        payload = request_item["payload"]
        url = self.services[tool]["generate_endpoint"]
        
        print(f"[GeneratorAgent] Calling {tool} API...")
        try:
            # Increase timeout for generation models
            # DiffGui is very slow (~2.5 min/mol), so we need a very long timeout
            # 20 mols * 3 min = 60 min = 3600s. Let's set to 2 hours to be safe.
            timeout = 7200 
            resp = requests.post(url, json=payload, timeout=timeout) 
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            raise RuntimeError(f"API call failed: {e}")

    def _prepare_payload(self, tool: str, task_params: Dict, target_data: Dict, n_samples: int) -> Dict:
        """Prepares the JSON payload for the API call, dynamically selecting modes based on context."""
        pdb_path = target_data.get("cleaned_pdb") or target_data.get("original_pdb")
        ref_ligand = target_data.get("ref_ligand_path")
        
        # Extract context from task parameters
        # IntentAgent or ConditionAgent should populate these based on user request
        gen_mode = task_params.get("generation_mode", "de_novo") # de_novo, linker_design, scaffold_decoration, optimization
        fragments_path = task_params.get("fragments_path")
        scaffold_path = task_params.get("scaffold_path")
        guidance = task_params.get("guidance", {})

        if not pdb_path:
            print(f"[GeneratorAgent] {tool} requires a PDB file.")
            return None

        if tool == "DiffSBDD":
            ckpt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "DiffSBDD", "checkpoints", "crossdocked_fullatom_cond.ckpt")
            
            payload = {
                "checkpoint": ckpt_path,
                "pdbfile": pdb_path,
                "n_samples": n_samples,
                "sanitize": True
            }
            
            # Mode Selection
            if gen_mode == "linker_design" and fragments_path:
                payload["mode"] = "inpaint"
                payload["fix_atoms"] = fragments_path
                # Inpainting usually requires a reference frame, often the fragments themselves or the original ligand
                payload["ref_ligand"] = fragments_path 
            elif gen_mode == "optimization" and ref_ligand:
                payload["mode"] = "optimize"
                payload["ref_ligand"] = ref_ligand
            else:
                # Default: De Novo
                if ref_ligand:
                    payload["ref_ligand"] = ref_ligand
                else:
                    # Fallback to pocket residues from fpocket
                    pocket_file = target_data.get("pocket_pdb_path")
                    
                    if not pocket_file:
                        pdb_name = os.path.basename(pdb_path)
                        base_name = os.path.splitext(pdb_name)[0]
                        # Try multiple potential output locations for fpocket
                        possible_dirs = [
                            os.path.join(os.path.dirname(pdb_path), f"{base_name}_out", "pockets"),
                            os.path.join(os.path.dirname(pdb_path), f"{pdb_name}_out", "pockets")
                        ]
                        
                        for p_dir in possible_dirs:
                            p_file = os.path.join(p_dir, "pocket1_atm.pdb")
                            if os.path.exists(p_file):
                                pocket_file = p_file
                                break
                    
                    if pocket_file:
                        resi_list = self._extract_residues_from_pdb(pocket_file)
                        if resi_list:
                            payload["resi_list"] = resi_list.split()
                        else:
                            return None
                    else:
                        # If no pocket info, try to use reference ligand from original pdb if available
                        # But we don't have it easily.
                        # DiffSBDD requires EITHER resi_list OR ref_ligand.
                        
                        # Last resort: Try to infer pocket from the protein PDB itself (center of mass?) 
                        # or just fail.
                        
                        # Let's try to supply a dummy ref_ligand if we have a file that might contain one?
                        # Or if the PDB file has a ligand inside?
                        
                        # For now, let's try to find a ligand in the input PDB to use as reference center
                        # This is a common use case: PDB has ligand, we want to redesign it.
                        if "pdb_path" in payload:
                            # Use the input PDB itself as potential source of reference ligand
                            # We can pass the PDB path as ref_ligand if the model supports extracting it,
                            # but DiffSBDD expects a separate file or a selection string.
                            
                            # Hack: Use a default residue list if nothing else works? No, that's bad.
                            
                            # If we are in this block, it means:
                            # 1. No explicit ref_ligand provided.
                            # 2. No fpocket output found.
                            
                            print("[GeneratorAgent] DiffSBDD: No reference ligand or pocket found. Attempting to auto-detect pocket center...")
                            # We can't easily auto-detect without external tools here.
                            # BUT, we can try to look for HETATM in the pdb file and use that as ref_ligand
                            
                            pass 
                        
                        print("[GeneratorAgent] DiffSBDD: No reference ligand or pocket found.")
                        return None
            return payload

        elif tool == "DecompDiff":
            # Select prior mode based on available reference
            prior_mode = "beta_prior" # Default (no prior info)
            if ref_ligand:
                prior_mode = "ref_prior"
            
            # If user specifically asks for subpocket exploration (advanced)
            if gen_mode == "subpocket_exploration":
                prior_mode = "subpocket"

            payload = {
                "pdb_path": pdb_path,
                "num_samples": n_samples,
                "prior_mode": prior_mode
            }
            return payload

        elif tool == "MiDi":
            # MiDi is typically conditional on PDB
            payload = {
                "pdb_path": pdb_path,
                "num_samples": n_samples,
                "mode": "conditional"
            }
            return payload

        elif tool == "GenMol":
            # Map high-level intent to GenMol specific tasks
            # API expects: denovo, linker_design, scaffold_morphing, motif_extension, scaffold_decoration, superstructure_generation
            
            gm_task = "denovo"
            fragment_smiles = None
            
            if gen_mode == "linker_design" and fragments_path:
                gm_task = "linker_design"
                fragment_smiles = self._read_smiles_from_file(fragments_path)
            elif gen_mode == "scaffold_decoration" and scaffold_path:
                gm_task = "scaffold_decoration"
                fragment_smiles = self._read_smiles_from_file(scaffold_path)
            elif gen_mode == "optimization":
                # Map optimization to scaffold morphing if reference exists
                if ref_ligand:
                    gm_task = "scaffold_morphing"
                    fragment_smiles = self._read_smiles_from_file(ref_ligand)
                elif fragments_path:
                    gm_task = "scaffold_morphing"
                    fragment_smiles = self._read_smiles_from_file(fragments_path)
            
            payload = {
                "task": gm_task,
                "num_samples": n_samples
            }
            
            if fragment_smiles:
                payload["fragment"] = fragment_smiles
                
            # Add optional parameters from task_params if present
            # Allow user/agent to override defaults
            if "randomness" in task_params:
                payload["randomness"] = float(task_params["randomness"])
            if "softmax_temp" in task_params:
                payload["softmax_temp"] = float(task_params["softmax_temp"])
            if "gamma" in task_params:
                payload["gamma"] = float(task_params["gamma"])
            
            return payload

        elif tool == "DiffGui":
            # Select mode based on input availability
            dg_mode = "denovo"
            if fragments_path:
                dg_mode = "frag_cond"
            
            # Extract constraints
            constraints = task_params.get("constraints", {})
            
            # Determine guidance type and weight based on constraints
            guidance_type = "uncertainty"
            guidance_weight = 1.e-4
            
            # Priority: QED > SA > LogP > TPSA > Affinity
            if "qed" in constraints or "qed_min" in constraints:
                guidance_type = "qed"
                guidance_weight = 10.0 # Strong guidance for QED
                if "qed_min" in constraints:
                    guidance["qed"] = float(constraints["qed_min"])
                elif "qed" in constraints:
                    guidance["qed"] = float(constraints["qed"])
            elif "sa" in constraints or "sa_max" in constraints:
                guidance_type = "sa"
                guidance_weight = 1.0
                if "sa_max" in constraints:
                    guidance["sa"] = float(constraints["sa_max"])
                elif "sa" in constraints:
                    guidance["sa"] = float(constraints["sa"])
            elif "logp" in constraints:
                guidance_type = "logp"
                guidance_weight = 1.0
                guidance["logp"] = float(constraints["logp"])
            elif "tpsa" in constraints:
                guidance_type = "tpsa"
                guidance_weight = 1.0
                guidance["tpsa"] = float(constraints["tpsa"])
            elif "affinity" in constraints or "docking_score" in constraints:
                guidance_type = "aff"
                guidance_weight = 10.0
            
            # Flatten guidance parameters
            guidance_params = {
                "logp": guidance.get("logp", 2.0),
                "tpsa": guidance.get("tpsa", 100.0),
                "sa": guidance.get("sa", 1.0),
                "qed": guidance.get("qed", 0.8),
                "aff": guidance.get("aff", 12.0),
                "gui_strength": guidance.get("gui_strength", 3.0)
            }

            # Ensure num_mols is an integer and within limits
            # DiffGui is slow, so we limit to 20 molecules max to avoid timeouts
            try:
                num_mols = int(n_samples)
                if num_mols > 20:
                    print(f"[GeneratorAgent] DiffGui: Limiting num_mols to 20 due to slow generation speed.")
                    num_mols = 20
            except (ValueError, TypeError):
                num_mols = 10

            # DiffGui requires a pocket PDB, not a full protein PDB
            # Try to find a pocket file if the provided pdb_path seems to be a full protein
            final_pocket_path = target_data.get("pocket_pdb_path")
            
            if not final_pocket_path:
                # Simple heuristic: check if "pocket" is in the filename
                final_pocket_path = pdb_path
                if "pocket" not in os.path.basename(pdb_path).lower():
                    pdb_name = os.path.basename(pdb_path)
                    base_name = os.path.splitext(pdb_name)[0]
                    possible_dirs = [
                        os.path.join(os.path.dirname(pdb_path), f"{base_name}_out", "pockets"),
                        os.path.join(os.path.dirname(pdb_path), f"{pdb_name}_out", "pockets")
                    ]
                    for p_dir in possible_dirs:
                        p_file = os.path.join(p_dir, "pocket1_atm.pdb")
                        if os.path.exists(p_file):
                            final_pocket_path = p_file
                            print(f"[GeneratorAgent] DiffGui: Using detected pocket file: {final_pocket_path}")
                            break
            else:
                print(f"[GeneratorAgent] DiffGui: Using pre-calculated pocket file: {final_pocket_path}")
            
            payload = {
                "pocket_pdb_path": final_pocket_path,
                "num_mols": num_mols,
                "batch_size": 8,
                "gen_mode": dg_mode,
                "ligand_sdf_path": ref_ligand if ref_ligand else "None",
                "frag_sdf_path": fragments_path if fragments_path else "None",
                "guidance_type": guidance_type,
                "guidance_weight": guidance_weight,
                **guidance_params
            }
            return payload
            
        return None

    def _extract_residues_from_pdb(self, pdb_path: str) -> str:
        """Extracts unique residue IDs (Chain:ResNum) from a PDB file."""
        residues = set()
        try:
            with open(pdb_path, 'r') as f:
                for line in f:
                    if line.startswith("ATOM"):
                        chain = line[21]
                        res_seq = line[22:26].strip()
                        residues.add(f"{chain}:{res_seq}")
            return " ".join(list(residues))
        except Exception:
            return None

    def _read_smiles_from_file(self, file_path: str) -> Optional[str]:
        """Reads the first molecule from a file (SDF/PDB/SMI) and returns its SMILES."""
        if not file_path or not os.path.exists(file_path):
            return None
            
        try:
            ext = os.path.splitext(file_path)[1].lower()
            mol = None
            
            if ext == ".sdf":
                suppl = Chem.SDMolSupplier(file_path)
                mol = suppl[0] if len(suppl) > 0 else None
            elif ext == ".pdb":
                mol = Chem.MolFromPDBFile(file_path)
            elif ext == ".smi" or ext == ".smiles":
                with open(file_path, "r") as f:
                    content = f.read().strip().split()[0]
                    return content # Assume file contains SMILES string
            
            if mol:
                return Chem.MolToSmiles(mol)
        except Exception as e:
            print(f"[GeneratorAgent] Error reading SMILES from {file_path}: {e}")
            
        return None

# LangGraph Node Wrapper
def generator_agent_node(state: AgentState) -> Dict[str, Any]:
    agent = GeneratorAgent()
    return agent.run(state)
