import os
import json
import requests
import numpy as np
import subprocess
import shutil
import time
from typing import Dict, Any, List, Optional, Tuple
from langchain_core.messages import HumanMessage, SystemMessage
from agent.base_agent import BaseAgent
from agent.state import AgentState
from agent.RAGAgent import RAGAgent

class TargetAgent(BaseAgent):
    def __init__(self):
        super().__init__(agent_name="TargetAgent")
        self.rag_agent = RAGAgent()
        # Define workspace for target data
        self.work_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed_targets")
        os.makedirs(self.work_dir, exist_ok=True)
        
        # Register available internal tools (mapping KG tool names to methods)
        # NOTE: TargetAgent should ONLY prepare data. It should NOT run generation models.
        self.tool_implementations = {
            "ValidatePDB": self.validate_pdb_id,
            "DownloadPDB": self.download_pdb,
            "FetchMetadata": self.fetch_target_metadata,
            "PDBFixer": self.run_pdbfixer,
            "CleanPDB": self.clean_pdb,
            "ExtractLigand": self.extract_ligand_from_pdb,
            "RDKit": self.calculate_center_from_sdf,
            "fpocket": self.run_fpocket,
            "CalculateCenterFromResidues": self.calculate_center_from_residues,
            "Finish": None # Special action
        }
        
        # Define tool descriptions for the LLM
        self.tool_descriptions = {
            "ValidatePDB": "Validates if a PDB ID exists and contains protein polymers. Args: pdb_id",
            "DownloadPDB": "Downloads a PDB file from RCSB. Args: pdb_id",
            "FetchMetadata": "Fetches biological metadata (title, organism, function) for a PDB ID. Args: pdb_id",
            "PDBFixer": "Repairs and cleans a PDB file (adds missing atoms/residues, removes water/ligands). Returns path to fixed PDB. Args: pdb_path",
            "CleanPDB": "Simple cleaning of PDB file (removes water/ligands) without fixing structure. Use if PDBFixer fails. Args: pdb_path",
            "ExtractLigand": "Extracts the largest co-crystallized ligand from a PDB file. Returns path to ligand SDF or None if not found. Args: pdb_path",
            "RDKit": "Calculates the geometric center of a ligand molecule file (SDF/PDB). Returns [x, y, z]. Args: sdf_path",
            "fpocket": "Detects binding pockets in a protein PDB using fpocket. Returns the center [x, y, z] of the largest pocket. Args: pdb_path",
            "CalculateCenterFromResidues": "Calculates the center of mass for specific residues. Args: pdb_path, resi_list (e.g. 'A:100 B:200')",
            "Finish": "Call this when all required data (cleaned PDB, pocket info, etc.) is prepared. Args: none"
        }

    def _get_kg_tools(self) -> List[Dict]:
        """Query KG for tools available to TargetAgent."""
        # Filter out generation tools if they accidentally appear in TargetAgent's KG definition
        tools = self.kg_loader.query_agent_tools("TargetAgent")
        return tools

    def _read_api_reference(self) -> str:
        """Reads the API Reference to understand model requirements using RAGAgent."""
        try:
            query = "What are the input file requirements (PDB, SDF, etc.) for all generation models?"
            context = self.rag_agent.retrieve(query, k=3)
            if context and "Knowledge base not initialized" not in context:
                return context
        except Exception:
            pass
        return "No API Reference available."

    def _format_history(self, history):
        history_str = ""
        for item in history:
            history_str += f"Step {item['step']}:\n"
            history_str += f"  Thought: {item.get('thought')}\n"
            history_str += f"  Action: {item.get('tool')} {item.get('args')}\n"
            history_str += f"  Result: {item.get('result')}\n\n"
        return history_str

    def run(self, state: AgentState) -> Dict[str, Any]:
        """
        Executes the target preparation workflow using a ReAct (Reason-Act) loop.
        The Agent dynamically decides which tools to use based on the task parameters,
        API reference, and the results of previous actions.
        """
        task_params = state.get("task_params", {})
        results = state.get("results", {})
        
        # 1. Context Gathering
        api_ref = self._read_api_reference()
        kg_tools = self._get_kg_tools()
        
        # 2. Execution State
        target_data = {} # Accumulates outputs (paths, centers)
        history = []     # Log of (Thought, Action, Result)
        
        max_steps = 10
        step = 0
        
        print(f"[TargetAgent] Starting ReAct loop for task: {task_params}")
        
        while step < max_steps:
            step += 1
            
            # A. Construct Prompt (SOP-Driven)
            sop_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sop", "TargetAgent_SOP.md")
            try:
                with open(sop_path, 'r') as f:
                    sop_content = f.read()
            except FileNotFoundError:
                sop_content = "No SOP file found. Proceed with standard tools."

            # We replace the old _construct_react_prompt logic with one that injects SOP
            # For simplicity, we can modify _construct_react_prompt or just build it here.
            # Let's modify _construct_react_prompt to accept sop_content if we want, 
            # OR just override it here.
            
            # Let's adhere to the BaseAgent pattern if possible, but for now we inject SOP directly into system message
            
            prompt = f"""
{sop_content}

Current Execution State:
- Task Parameters: {json.dumps(task_params, indent=2)}
- API Reference Summary: {api_ref}
- Available Tools (from KG): {[t['name'] for t in kg_tools]}
- Execution History:
{self._format_history(history)}

Current Data Context:
{json.dumps(target_data, indent=2)}

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
            
            # B. Call LLM
            try:
                messages = [SystemMessage(content=prompt)]
                response = self.model.invoke(messages)
                content = response.content.strip()
            except Exception as e:
                print(f"[TargetAgent] LLM Invocation Failed: {e}")
                # Wait and retry once
                time.sleep(5)
                try:
                    print("[TargetAgent] Retrying LLM invocation...")
                    response = self.model.invoke(messages)
                    content = response.content.strip()
                except Exception as e2:
                    print(f"[TargetAgent] LLM Retry Failed: {e2}")
                    history.append({"step": step, "error": "LLM Error", "content": str(e2)})
                    break
            
            # C. Parse Action
            action = self._parse_action(content)
            
            if not action:
                print(f"[TargetAgent] Failed to parse action from: {content}")
                history.append({"step": step, "error": "Parse Error", "content": content})
                continue
                
            tool_name = action.get("tool") or action.get("action")
            args = action.get("args") or action.get("action_input") or {}
            thought = action.get("thought", "")
            
            print(f"[TargetAgent] Step {step}")
            print(f"  Thought: {thought}")
            print(f"  Tool: {tool_name} Args: {args}")
            
            # D. Check Finish
            if tool_name == "Finish":
                print("[TargetAgent] Agent decided to finish.")
                break
            
            # E. Execute Tool
            result = self._execute_tool(tool_name, args, target_data)
            
            # F. Update History
            history.append({
                "step": step,
                "thought": thought,
                "tool": tool_name,
                "args": args,
                "result": str(result)
            })
            
        results["target_preparation"] = target_data
        return {
            "current_agent": "TargetAgent",
            "results": results,
            "task_params": task_params
        }

    def _construct_react_prompt(self, task_params, api_ref, kg_tools, history, target_data):
        # Format tool descriptions
        tools_desc_str = json.dumps(self.tool_descriptions, indent=2)
        
        # Format KG tools for context (Reasoning Layer)
        kg_context_lines = []
        for t in kg_tools:
            name = t.get('name', 'Unknown')
            func = t.get('function', '')
            cat = t.get('category', '')
            kg_context_lines.append(f"- {name} ({cat}): {func}")
        kg_context_str = "\n".join(kg_context_lines)
        
        history_str = ""
        for item in history:
            history_str += f"Step {item['step']}:\n"
            history_str += f"  Thought: {item.get('thought')}\n"
            history_str += f"  Action: {item.get('tool')} {item.get('args')}\n"
            history_str += f"  Result: {item.get('result')}\n\n"
            
        current_data_str = json.dumps(target_data, indent=2)
        
        prompt = f"""
You are the TargetAgent, an expert in structural biology and drug discovery.
Your goal is to prepare protein target data for downstream generation models.

### CRITICAL INSTRUCTION:
**You are a DATA PREPARATION agent only.**
- You do **NOT** have access to generation models (like DiffSBDD, DrugDiff, etc.).
- You **CANNOT** generate molecules.
- Your job is to **PREPARE** the input files (PDB, pocket center, reference ligand) that those models will need later.
- Once you have prepared the PDB and pocket information, you **MUST** call the `Finish` tool. Do not try to run the generation model yourself.

### Inputs:
- **Task Parameters**: {json.dumps(task_params, indent=2)}
- **Downstream Model Requirements** (READ ONLY - DO NOT EXECUTE):
{api_ref}
*Use the above requirements to know WHAT data to prepare (e.g., if DiffSBDD needs a pocket center, you must calculate it).*

### Knowledge Graph Insights (Domain Capabilities):
The Knowledge Graph defines the theoretical capabilities available to you. Use this for high-level planning.
{kg_context_str}

### Available Tools (Executable Actions):
These are the ONLY tools you can call.
{tools_desc_str}

### Current State:
- **Collected Data**: {current_data_str}
- **Execution History**:
{history_str}

### In-Context Examples (Few-Shot Learning):
Example 1:
Task: "Prepare target 1ABC for DiffSBDD."
Step 1:
{{
  "thought": "I need to download the PDB file and fetch its metadata.",
  "tool": "FetchMetadata",
  "args": {{ "pdb_id": "1abc" }}
}}
Step 2:
{{
  "thought": "Now I will download the PDB structure.",
  "tool": "DownloadPDB",
  "args": {{ "pdb_id": "1abc" }}
}}
Step 3:
{{
  "thought": "Now I need to clean the PDB file to remove water and non-protein atoms.",
  "tool": "PDBFixer",
  "args": {{ "pdb_path": "/path/to/1abc.pdb" }}
}}
Step 4:
{{
  "thought": "DiffSBDD requires a reference ligand or pocket center. I will try to extract the ligand.",
  "tool": "ExtractLigand",
  "args": {{ "pdb_path": "/path/to/1abc.pdb" }}
}}
Step 5:
{{
  "thought": "I have the cleaned PDB, metadata, and the reference ligand. I am ready to hand off to the generator.",
  "tool": "Finish",
  "args": {{}}
}}

Example 2:
Task: "Find binding pocket for 2Z3H."
Step 1:
{{
  "thought": "I will download the PDB structure.",
  "tool": "DownloadPDB",
  "args": {{ "pdb_id": "2z3h" }}
}}
Step 2:
{{
  "thought": "To find the pocket, I will use fpocket as suggested by the KG.",
  "tool": "fpocket",
  "args": {{ "pdb_path": "/path/to/2z3h.pdb" }}
}}
Step 3:
{{
  "thought": "I have the pocket center. Task complete.",
  "tool": "Finish",
  "args": {{}}
}}
 
### Instructions:
1. **Analyze** the history and current data.
2. **Review** the 'Downstream Model Requirements' to see what data is missing (e.g. do I need a pocket center?).
3. **Select** the most appropriate tool from 'Available Tools'.
   - **NEVER** output a tool name that is not in 'Available Tools'.
   - If you think you need to run 'DiffSBDD', you are wrong. You should instead check if you have prepared the inputs for it, and if so, call 'Finish'.
4. **Pocket Center Calculation (MANDATORY)**:
   - Most downstream models (DiffSBDD, DiffGui, Docking) **REQUIRE** a pocket center [x, y, z].
   - If you have a PDB, you **MUST** try to calculate the pocket center before finishing.
   - Use `ExtractLigand` -> `RDKit` (if ligand exists) OR `fpocket` (if no ligand) to find the center.
   - Do NOT finish without a pocket center unless you have tried all methods and failed.
5. **Finish**: When you have prepared all the data required by the downstream model, call the `Finish` tool.

### Output Format:
Return a JSON object with "thought", "tool", and "args".
```json
{{
  "thought": "I need to download the PDB file first.",
  "tool": "DownloadPDB",
  "args": {{ "pdb_id": "1abc" }}
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

    def _execute_tool(self, tool_name, args, target_data):
        if tool_name not in self.tool_implementations:
            return f"Error: Tool {tool_name} not found."
            
        func = self.tool_implementations[tool_name]
        
        try:
            result = func(**args)
            
            # Side-effects: Update target_data based on tool success
            if tool_name == "DownloadPDB":
                target_data["original_pdb"] = result
            elif tool_name == "FetchMetadata":
                if isinstance(result, dict) and "error" not in result:
                    target_data["target_metadata"] = result
            elif tool_name == "PDBFixer":
                target_data["cleaned_pdb"] = result
            elif tool_name == "ExtractLigand":
                if result:
                    target_data["ref_ligand_path"] = result
            elif tool_name == "fpocket":
                if result is not None:
                    center, pocket_path = result
                    target_data["pocket_center"] = center.tolist() if isinstance(center, np.ndarray) else center
                    target_data["pocket_pdb_path"] = pocket_path
                    target_data["pocket_method"] = tool_name
            elif tool_name in ["RDKit", "CalculateCenterFromResidues"]:
                if result is not None:
                    # Convert numpy array to list for JSON serialization
                    center = result.tolist() if isinstance(result, np.ndarray) else result
                    target_data["pocket_center"] = center
                    target_data["pocket_method"] = tool_name
            
            return result
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"

    # --- Internal Tools ---

    def fetch_target_metadata(self, pdb_id: str) -> Dict[str, Any]:
        """Fetches metadata for a PDB ID from RCSB Data API."""
        try:
            url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                title = data.get("struct", {}).get("title", "Unknown Title")
                desc = data.get("struct", {}).get("pdbx_descriptor", "")
                
                # Try to get organism
                organism = "Unknown"
                if "rcsb_entity_source_organism" in data:
                    sources = data["rcsb_entity_source_organism"]
                    if sources and isinstance(sources, list):
                        organism = sources[0].get("ncbi_scientific_name", "Unknown")
                
                metadata = {
                    "pdb_id": pdb_id,
                    "title": title,
                    "description": desc,
                    "organism": organism,
                    "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                return metadata
            else:
                return {"error": f"Failed to fetch metadata: HTTP {response.status_code}"}
        except Exception as e:
            return {"error": f"Exception fetching metadata: {str(e)}"}

    def download_pdb(self, pdb_id: str) -> str:
        """Downloads a PDB file from RCSB."""
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        response = requests.get(url)
        if response.status_code != 200:
            raise ValueError(f"Failed to download PDB {pdb_id}")
        
        file_path = os.path.join(self.work_dir, f"{pdb_id}.pdb")
        with open(file_path, "w") as f:
            f.write(response.text)
        return file_path

    def clean_pdb(self, pdb_path: str, output_name: str = None) -> str:
        """
        Removes HETATM (waters, ligands) and keeps only ATOM records.
        Uses simple text processing to avoid heavy dependencies if Bio is missing,
        but tries Bio.PDB first.
        """
        if output_name is None:
            base_name = os.path.basename(pdb_path).split('.')[0]
            output_name = f"{base_name}_clean.pdb"
            
        output_path = os.path.join(self.work_dir, output_name)
        
        try:
            from Bio import PDB
            parser = PDB.PDBParser(QUIET=True)
            structure = parser.get_structure("target", pdb_path)
            
            class NotHetatm(PDB.Select):
                def accept_residue(self, residue):
                    # Keep standard residues (ATOM), discard HETATM (water, ligands)
                    # PDB.Select methods: accept_residue returns 1 to keep, 0 to discard
                    return 1 if PDB.is_aa(residue, standard=True) else 0

            io = PDB.PDBIO()
            io.set_structure(structure)
            io.save(output_path, select=NotHetatm())
            return output_path
            
        except ImportError:
            print("[TargetAgent] BioPython not found. Using text-based cleaning.")
            with open(pdb_path, "r") as f_in, open(output_path, "w") as f_out:
                for line in f_in:
                    if line.startswith("ATOM"):
                        f_out.write(line)
                    elif line.startswith("TER"):
                        f_out.write(line)
                    elif line.startswith("END"):
                        f_out.write(line)
            return output_path

    def calculate_center_from_residues(self, pdb_path: str, resi_list: str) -> np.ndarray:
        """
        Calculates center of mass of specified residues.
        resi_list format example: "A:105 B:200" (Chain:ResNum)
        """
        # Parse resi_list
        target_residues = []
        for item in resi_list.split():
            if ":" in item:
                chain, resnum = item.split(":")
                target_residues.append((chain, int(resnum)))
        
        coords = []
        try:
            from Bio import PDB
            parser = PDB.PDBParser(QUIET=True)
            structure = parser.get_structure("target", pdb_path)
            
            for model in structure:
                for chain in model:
                    for residue in chain:
                        res_id = residue.get_id()[1]
                        chain_id = chain.get_id()
                        if (chain_id, res_id) in target_residues:
                            for atom in residue:
                                coords.append(atom.get_coord())
            
            if not coords:
                raise ValueError("No matching residues found.")
            
            return np.mean(coords, axis=0)
            
        except ImportError:
            # Text based fallback (Much harder for specific residues, simplified here)
            # Assuming standard PDB columns
            with open(pdb_path, "r") as f:
                for line in f:
                    if line.startswith("ATOM"):
                        chain_id = line[21]
                        res_seq = int(line[22:26])
                        if (chain_id, res_seq) in target_residues:
                            x = float(line[30:38])
                            y = float(line[38:46])
                            z = float(line[46:54])
                            coords.append([x, y, z])
            
            if not coords:
                raise ValueError("No matching residues found (Text fallback).")
            return np.mean(coords, axis=0)

    def calculate_center_from_sdf(self, sdf_path: str) -> np.ndarray:
        """Calculates center of mass of a ligand from SDF."""
        coords = []
        with open(sdf_path, "r") as f:
            lines = f.readlines()
            # Simple SDF parser for coordinates block
            # Find atom block start
            # This is a very naive parser, assuming standard SDF format
            # Better to use RDKit if available
            try:
                from rdkit import Chem
                suppl = Chem.SDMolSupplier(sdf_path)
                mol = suppl[0]
                conf = mol.GetConformer()
                pos = conf.GetPositions()
                return np.mean(pos, axis=0)
            except ImportError:
                print("[TargetAgent] RDKit not found. Using text-based SDF parsing.")
                # Skip header
                # Look for counts line (4th line usually)
                # Then read atom lines
                # This is risky, but a fallback
                pass
        
        return np.array([0.0, 0.0, 0.0]) # Placeholder

    def calculate_center_of_mass(self, pdb_path: str) -> np.ndarray:
        """Calculates center of mass of the whole protein."""
        coords = []
        with open(pdb_path, "r") as f:
            for line in f:
                if line.startswith("ATOM"):
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
    def calculate_center_of_mass(self, pdb_path: str) -> np.ndarray:
        """Calculates the center of mass of a PDB file."""
        coords = []
        try:
            with open(pdb_path, 'r') as f:
                for line in f:
                    if line.startswith("ATOM") or line.startswith("HETATM"):
                        try:
                            x = float(line[30:38])
                            y = float(line[38:46])
                            z = float(line[46:54])
                            coords.append([x, y, z])
                        except ValueError:
                            continue
        except Exception as e:
            print(f"[TargetAgent] Error reading PDB for center calculation: {e}")
            return np.array([0.0, 0.0, 0.0])
            
        if not coords:
            return np.array([0.0, 0.0, 0.0])
        return np.mean(coords, axis=0)

    def extract_ligand_from_pdb(self, pdb_path: str, output_name: str = None) -> str:
        """
        Extracts the largest HETATM ligand from a PDB file and saves it as SDF/PDB.
        """
        if output_name is None:
            base_name = os.path.basename(pdb_path).split('.')[0]
            output_name = f"{base_name}_ligand.sdf"
            
        output_path = os.path.join(self.work_dir, output_name)
        
        try:
            from Bio import PDB
            parser = PDB.PDBParser(QUIET=True)
            structure = parser.get_structure("target", pdb_path)
            
            ligands = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        # Check if it's a HETATM and not water
                        if residue.id[0].startswith("H_") and residue.resname != "HOH":
                            ligands.append(residue)
            
            if not ligands:
                print("[TargetAgent] No ligands found in PDB.")
                return None
                
            # Select largest ligand by atom count
            largest_ligand = max(ligands, key=lambda r: len(r))
            print(f"[TargetAgent] Found ligand: {largest_ligand.resname} (Chain {largest_ligand.get_parent().id})")
            
            # Save as PDB first (BioPython handles PDB easily)
            io = PDB.PDBIO()
            class LigandSelect(PDB.Select):
                def accept_residue(self, residue):
                    return 1 if residue == largest_ligand else 0
            
            temp_pdb = output_path.replace(".sdf", ".pdb")
            io.set_structure(structure)
            io.save(temp_pdb, select=LigandSelect())
            
            # Convert to SDF if RDKit available
            try:
                from rdkit import Chem
                mol = Chem.MolFromPDBFile(temp_pdb)
                if mol:
                    writer = Chem.SDWriter(output_path)
                    writer.write(mol)
                    writer.close()
                    return output_path
                else:
                    return temp_pdb # Fallback to PDB format
            except ImportError:
                return temp_pdb

        except ImportError:
            print("[TargetAgent] BioPython not found. Cannot extract ligand.")
            return None

    def validate_pdb_id(self, pdb_id: str) -> Dict[str, Any]:
        """Queries RCSB API to validate if PDB ID exists and contains protein."""
        url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.lower()}"
        try:
            response = requests.get(url)
            if response.status_code == 404:
                return {"valid": False, "reason": "PDB ID not found"}
            
            data = response.json()
            # Check for protein polymer
            polymer_count = data.get("rcsb_entry_info", {}).get("polymer_entity_count_protein", 0)
            
            if polymer_count == 0:
                return {"valid": False, "reason": "Entry does not contain protein polymers (might be DNA/RNA or Ligand only)"}
            
            title = data.get("struct", {}).get("title", "Unknown Protein")
            return {"valid": True, "info": data, "title": title}
        except Exception as e:
            # If API fails, we might warn but proceed (soft fail)
            print(f"[TargetAgent] Validation API failed: {e}")
            return {"valid": True, "warning": "Could not validate PDB ID via API"}

    def run_fpocket(self, pdb_path: str) -> Tuple[np.ndarray, str]:
        """
        Runs fpocket to detect binding pockets and returns the center of the largest pocket.
        """
        # Check if fpocket is installed
        if shutil.which("fpocket") is None:
            print("[TargetAgent] fpocket executable not found in PATH.")
            return None

        print(f"[TargetAgent] Running fpocket on {pdb_path}...")
        try:
            # Run fpocket
            # fpocket output goes to <pdb_filename_without_ext>_out directory
            subprocess.run(["fpocket", "-f", pdb_path], check=True, capture_output=True)
            
            pdb_name = os.path.basename(pdb_path)
            base_name = os.path.splitext(pdb_name)[0]
            output_dir = os.path.join(os.path.dirname(pdb_path), f"{base_name}_out")
            
            if not os.path.exists(output_dir):
                # Fallback: sometimes it might use the full name?
                # Let's check if it used the full name
                alt_output_dir = os.path.join(os.path.dirname(pdb_path), f"{pdb_name}_out")
                if os.path.exists(alt_output_dir):
                    output_dir = alt_output_dir
                else:
                    print(f"[TargetAgent] fpocket output directory not found: {output_dir}")
                    return None
                
            # Find the first pocket (pocket1_atm.pdb is usually the highest ranked/largest)
            pocket1_path = os.path.join(output_dir, "pockets", "pocket1_atm.pdb")
            
            if os.path.exists(pocket1_path):
                print(f"[TargetAgent] Found best pocket: {pocket1_path}")
                # Calculate center of this pocket using existing method
                center = self.calculate_center_of_mass(pocket1_path)
                # Save pocket path to target_data for downstream use (e.g. DiffGui)
                # We return it as part of the tuple, but let's ensure it's clear
                return center, pocket1_path
            else:
                print("[TargetAgent] No pockets found by fpocket.")
                return None
                
        except subprocess.CalledProcessError as e:
            print(f"[TargetAgent] fpocket execution failed: {e}")
            return None
        except Exception as e:
            print(f"[TargetAgent] Error running fpocket: {e}")
            return None

    def run_pdbfixer(self, pdb_path: str, output_name: str = None) -> str:
        """
        Uses PDBFixer to fix missing residues, atoms, and hydrogens in the PDB file.
        """
        if output_name is None:
            base_name = os.path.basename(pdb_path).split('.')[0]
            output_name = f"{base_name}_fixed.pdb"
            
        output_path = os.path.join(self.work_dir, output_name)
        
        print(f"[TargetAgent] Running PDBFixer on {pdb_path}...")
        
        try:
            enable_pdbfixer = os.getenv("TARGETAGENT_PDBFIXER_ENABLE", "1").strip().lower() not in {"0", "false", "no", "off"}
            if not enable_pdbfixer:
                return self.clean_pdb(pdb_path, output_name)

            add_h = os.getenv("TARGETAGENT_PDBFIXER_ADD_H", "1").strip().lower() not in {"0", "false", "no", "off"}
            try:
                ph = float(os.getenv("TARGETAGENT_PDBFIXER_PH", "7.0"))
            except Exception:
                ph = 7.0

            try:
                timeout_s = int(os.getenv("TARGETAGENT_PDBFIXER_TIMEOUT_S", "0"))
            except Exception:
                timeout_s = 0

            if timeout_s and timeout_s > 0:
                import sys
                import tempfile

                with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as tmp_out:
                    tmp_out_path = tmp_out.name

                script = f"""
from pdbfixer import PDBFixer
from openmm.app import PDBFile

fixer = PDBFixer(filename={pdb_path!r})
fixer.findMissingResidues()
fixer.findMissingAtoms()
fixer.addMissingAtoms()
if {add_h!r}:
    fixer.addMissingHydrogens({ph!r})
with open({tmp_out_path!r}, "w") as f:
    PDBFile.writeFile(fixer.topology, fixer.positions, f)
print("OK")
"""
                try:
                    proc = subprocess.run(
                        [sys.executable, "-c", script],
                        capture_output=True,
                        text=True,
                        timeout=timeout_s,
                        check=False,
                    )
                except subprocess.TimeoutExpired:
                    try:
                        os.remove(tmp_out_path)
                    except Exception:
                        pass
                    raise TimeoutError(f"PDBFixer timeout after {timeout_s}s")

                if proc.returncode != 0:
                    try:
                        os.remove(tmp_out_path)
                    except Exception:
                        pass
                    err = (proc.stderr or proc.stdout or "").strip()
                    raise RuntimeError(f"PDBFixer subprocess failed: {err[:300]}")

                shutil.move(tmp_out_path, output_path)
                print(f"[TargetAgent] PDBFixer completed. Saved to {output_path}")
                return output_path

            from pdbfixer import PDBFixer
            from openmm.app import PDBFile

            fixer = PDBFixer(filename=pdb_path)
            print("[TargetAgent] PDBFixer: findMissingResidues()")
            fixer.findMissingResidues()
            print("[TargetAgent] PDBFixer: findMissingAtoms()")
            fixer.findMissingAtoms()
            print("[TargetAgent] PDBFixer: addMissingAtoms()")
            fixer.addMissingAtoms()
            if add_h:
                print(f"[TargetAgent] PDBFixer: addMissingHydrogens(pH={ph})")
                fixer.addMissingHydrogens(ph)
            with open(output_path, 'w') as f:
                PDBFile.writeFile(fixer.topology, fixer.positions, f)

            print(f"[TargetAgent] PDBFixer completed. Saved to {output_path}")
            return output_path
            
        except ImportError:
            print("[TargetAgent] PDBFixer or OpenMM not found. Falling back to simple cleaning.")
            return self.clean_pdb(pdb_path, output_name)
        except Exception as e:
            print(f"[TargetAgent] PDBFixer failed: {e}")
            raise

# LangGraph Node Wrapper
def target_agent_node(state: AgentState) -> Dict[str, Any]:
    agent = TargetAgent()
    return agent.run(state)
