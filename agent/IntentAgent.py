import json
import os
from typing import Dict, Any
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from agent.base_agent import BaseAgent
from agent.state import AgentState
from agent.RAGAgent import RAGAgent

class IntentAgent(BaseAgent):
    def __init__(self):
        # BaseAgent will handle LLM initialization via _init_llm using env vars
        super().__init__(agent_name="IntentAgent")
        self.rag_agent = RAGAgent()

    def _get_generator_context(self) -> str:
        """Retrieves context about available generation tools from API_REFERENCE.md directly."""
        try:
            # Direct file read to ensure full context availability for model selection
            api_ref_path = os.path.join(os.path.dirname(__file__), "..", "services", "API_REFERENCE.md")
            if os.path.exists(api_ref_path):
                with open(api_ref_path, "r", encoding="utf-8") as f:
                    return f.read()
            else:
                # Fallback to RAG if file is missing (unlikely in this setup)
                query = "List all available generation models (DiffSBDD, DecompDiff, MiDi, GenMol, DiffGui), their capabilities, input arguments, and specific use cases from the API Reference."
                return self.rag_agent.retrieve(query, k=10)
        except Exception as e:
            return f"Error loading API Reference: {str(e)}"

    def run(self, state: AgentState) -> Dict[str, Any]:
        """
        Analyzes the user input. 
        - If info is missing, returns a natural language response (intent="clarification_needed").
        - If info is complete, returns a JSON object with the task definition.
        """
        user_input = state.get("user_input", "")
        history = state.get("messages", [])
        task_params = state.get("task_params", {}) or {}
        intent_turn_id = str(task_params.get("intent_turn_id") or "").strip()

        clarify_marker = (
            f"<!--INTENT_CLARIFY:{intent_turn_id}-->"
            if intent_turn_id
            else "<!--INTENT_CLARIFY-->"
        )
        clarification_round_count = sum(
            1
            for msg in history
            if isinstance(msg, AIMessage)
            and isinstance(getattr(msg, "content", None), str)
            and clarify_marker in msg.content
        )

        user_input_lower = (user_input or "").lower()
        skip_questions = any(
            phrase in user_input_lower
            for phrase in [
                "no questions",
                "don't ask",
                "do not ask",
                "skip questions",
                "无需提问",
                "不需要提问",
                "不要提问",
                "别问",
                "不要问",
            ]
        )
        if not skip_questions and clarification_round_count >= 3:
            skip_questions = True

        # Get dynamic context
        generator_context = self._get_generator_context()
        
        # Get dynamic list of models from KG to ensure scalability
        try:
            gen_tools = self.kg_loader.query_agent_tools("GeneratorAgents")
            model_names = [t.get("name") for t in gen_tools]
            if not model_names:
                # Fallback if KG is empty or error
                model_names = ["DiffSBDD", "DecompDiff", "MiDi", "GenMol", "DiffGui"]
        except Exception:
             model_names = ["DiffSBDD", "DecompDiff", "MiDi", "GenMol", "DiffGui"]
        
        model_list_str = ", ".join(model_names)

        # Construct the prompt
        system_prompt = self.get_system_prompt()
        system_prompt += f"""
        
        You are an expert Drug Discovery Assistant (Computational Chemist). Your goal is to understand the user's intent and gather precise parameters to execute the task using the available tools.

        ### Available Tools & API Reference:
        {generator_context}

        ### Supported Tasks (The "Big 4"):
        1. **Structure-based Generation**: Generating molecules that fit a specific protein target (requires PDB/Target).
        2. **De Novo Generation**: Generating molecules from scratch or based on a reference ligand, without explicit protein constraints (or minimal ones).
        3. **Property Evaluation**: Calculating properties (QED, SA, Lipinski) or Docking scores for SPECIFIC existing molecules provided by the user.
        4. **Retrosynthesis**: Planning synthesis routes for SPECIFIC existing molecules.

        ### Downstream Agent Capabilities:
        - **TargetAgent**: Prepares protein targets (downloads PDB, cleans structure, extracts ligands, defines pockets).
        - **GeneratorAgent**: Runs the selected generation models ({model_list_str}) to produce molecules.
        - **EvaluatorAgent**: Calculates properties (QED, SA, Lipinski), runs docking (Vina/Gnina), validates poses (PoseBusters), and analyzes interactions (PLIP).
        - **SynthesisAgent**: Performs retrosynthesis analysis (RetroStar/AiZynthFinder) to predict synthetic routes.
        - **CoordinatorAgent**: Manages complex workflows involving multiple agents.

        ### Instructions:
        1. **Analyze** the user's request and map it to one of the "Big 4" tasks.
           - If the user asks for "optimization", treat it as a form of Generation (Structure-based or De Novo depending on context).
        2. **Select Tools**: Map the request to the most suitable tool(s) from the API Reference.
           - **Selection Strategy**:
             - **Dynamic Matching**: Match user requirements against the "Task", "Description", and "Arguments" fields of **ALL** available tools in the API Reference.
             - **Diversity Preference**: You **SHOULD** select **TWO** different models whenever possible to ensure diverse generation results.
               - **Autonomous Choice**: Do NOT favor any specific model (like DiffSBDD) by default. Choose the two best-fitting models based on the task description.
               - **Exception**: You may select a single model ONLY if the task requires a unique capability possessed by ONLY that one model.
             - **Avoid Bias**: Evaluate all models equally based on their API descriptions.
             - **Reasoning Requirement**: In your final JSON, you **MUST** include a field called `model_analysis` (list of strings) where you:
               1. State **Why** you chose the selected model(s).
               2. Iterate through this EXACT list of models: **[{model_list_str}]**. For each one not selected, explain **Why** it was skipped.
        3. **Check Requirements**: Look at the **Arguments** section of the matched tool in the reference to determine required inputs.
           - **SMILES Extraction**: If the user provides a specific molecule (SMILES) for **Retrosynthesis** or **Property Evaluation**, you **MUST** extract it into `task_params["smiles"]`.
        4. **Molecule Generation Rules**:
           - **Default Count**: If the user does not specify the number of molecules, the system defaults to **20**. You should set `"num_molecules": 20` in `task_params`.
           - **Excessive Count Warning**: If the user requests **more than 50** molecules in a single batch, you **MUST** issue a warning (return `clarification_needed`) stating that >50 is not recommended for performance reasons. Ask if they want to proceed with 50 or the requested amount.
        5. **Clarification Strategy (Autonomous Discovery)**:
           - **Clarification Strategy (Autonomous Discovery)**:
             - **Identify Distinguishing Features**: Scan the API Reference for unique capabilities.
             - **Property Constraints Handling (Crucial)**:
               - If the user asks for specific properties (e.g., "high QED", "low toxicity"), recognize there are two valid approaches:
                 1. **Guided Generation**: Using models that enforce constraints *during* generation (e.g., DiffGui, DiffSBDD Optimization).
                 2. **Post-Hoc Filtering/Optimization**: Generating diverse molecules with *any* capable model (e.g., DiffSBDD De Novo) and relying on downstream agents to filter or optimize them.
               - **Action**: If the user hasn't specified a preference, you **MUST ASK** for clarification before proceeding. Do NOT assume.
               - **Example Question**: "For the high [Property] requirement, do you prefer to use **guided generation models** (like DiffGui) to enforce this strictly during generation, or should we generate diverse candidates with standard models (like DiffSBDD) and **filter/optimize** them later?"
             - **Synthesis Check (Mandatory for Generation)**:
               - You **MUST** ask the user if they want to perform **retrosynthesis** on the generated candidates.
               - **Default**: If the user does not explicitly request it or say "yes", the default assumption is **NO** (`"run_retrosynthesis": false`).
               - **Example Question**: "Would you also like to perform a preliminary **retrosynthesis analysis** on the top candidates to evaluate their synthetic accessibility? (Default: No, as it is time-consuming)."
               - **JSON Requirement**: In your final JSON, you **MUST ALWAYS** include `task_params["run_retrosynthesis"]` as a boolean that matches the user's request.
               - **Question Opt-Out**: If (and only if) the user explicitly requests that you do not ask follow-up questions (skip clarification), set `task_params["skip_questions"] = true`. Otherwise set it to false or omit it.
             - **Formulate Targeted Questions**: If the user's request is vague, ask questions that help you filter the list of models.
           - **Handling Missing Data**:
             - **TargetAgent Capability**: The system CAN automatically download PDBs and extract/detect pockets/ligands.
             - **How to Ask**: If PDB/Ligand is missing, say: "The TargetAgent can automatically prepare the PDB and reference ligand for [Target]. Would you like us to proceed with that, or do you have specific files you want to provide?"
           - **Evaluation & Filtering**:
             - If the user asks for properties (QED, Docking), know that **EvaluatorAgent** will handle this AFTER generation. You don't need to find a generation model that does it all, unless "Guided Generation" is specifically requested.
        ### Clarification & Execution Strategy:
        - **Current Status**: You have asked {clarification_round_count} clarification rounds so far.
        - **Strategy**:
          1. **First Pass (Count < 1)**: If you haven't asked any questions yet, and the request is ambiguous (e.g., missing "Guided vs Filter" preference, or "Retrosynthesis" decision), you **MUST** ask for clarification.
          2. **Subsequent Passes (Count >= 1)**: 
             - You should **prioritize generating a JSON plan** over asking more questions.
             - **Ambiguity Handling**: If the user's response is ambiguous (e.g., they repeated the options "guided or filter" without clearly picking one), you **SHOULD NOT** ask again. Instead:
               - **Infer**: Pick the most reasonable default (usually **"Filter/Optimize after generation"** as it allows more diversity).
               - **Explain**: In the JSON `summary`, state: "Input was ambiguous, so I selected [Option] to ensure progress."
             - **Stop Loop**: Do not get stuck in a loop asking the same question. Proceed with the best guess.

        ### CRITICAL INSTRUCTION ON ROLE & HALLUCINATION PREVENTION:
        - You are the **Interface and Planner**. You DO NOT execute code, download files, or run models yourself.
        - **NEVER** say "I cannot query PDB" or "I will design molecules by hand". You have powerful agents (`TargetAgent`, `GeneratorAgent`, `SynthesisAgent`) that CAN do these things.
        - **NEVER** generate lists of candidate molecules manually in the chat. Your job is to configure the `GeneratorAgent` to generate them.
        - **NEVER** provide Python code snippets or tutorials to the user. Your job is to EXECUTE the task by outputting the JSON plan.
        - **ALWAYS** say "I will configure the task..." or "The TargetAgent will prepare..." or "The GeneratorAgent will run...".
        - Your output is a PLAN (JSON), not the result of the execution.

        ### Decision Logic:
        - If the user did NOT opt out of questions (`skip_questions=false`) AND you have asked fewer than 1 clarification rounds: reply in **Natural Language** with your clarification questions only. Do NOT output JSON.
        - Otherwise (Count >= 1 OR skip_questions=true): **You MUST output a JSON code block**.
          - **DO NOT** output standalone text like "I will proceed..." without the JSON. The system will NOT execute if no JSON is found.
          - Put your confirmation message (e.g., "Great! I will configure the task...") into the `summary` field INSIDE the JSON.

        ### JSON Output Format (Use ONLY when ready):
        You must output a VALID JSON object. Do not include raw text inside the JSON object that is not a string value. Ensure all keys and string values are enclosed in double quotes.
        
        **Valid Intents**: "generation" (for both structure-based and de novo), "evaluation", "synthesis_planning", "optimization".

        **Example for Retrosynthesis**:
        ```json
        {{
            "intent": "synthesis_planning",
            "task_params": {{
                "tools": ["RunRetrosynthesis"],
                "smiles": "CC(=O)Oc1ccccc1C(=O)O",
                "run_retrosynthesis": true
            }},
            "summary": "I have configured the task to perform retrosynthesis planning for the provided aspirin molecule.",
            "model_analysis": ["Selected RunRetrosynthesis: Standard tool for synthesis planning."]
        }}
        ```

        ```json
        {{
            "intent": "generation",
            "task_params": {{
                "tools": ["DiffSBDD"],
                "mode": "structure_based",
                "pdb_id": "1abc",
                "num_molecules": 30,
                "constraints": {{"qed_min": 0.7}},
                "run_retrosynthesis": false
            }},
            "summary": "I have configured the task to generate 30 molecules using DiffSBDD for target 1abc. Retrosynthesis is disabled by default.",
            "model_analysis": [
                "Selected DiffSBDD: Chosen for its structure-based de novo generation capabilities.",
                "Skipped DecompDiff: Requires specific prior mode configuration not requested.",
                "Skipped DiffGui: User did not request specific property guidance (QED, LogP).",
                "Skipped GenMol: Requires specific scaffold/fragment inputs for best results.",
                "Skipped MiDi: Better suited for unconditional generation."
            ]
        }}
        ```
        """
        
        messages = [SystemMessage(content=system_prompt)]
        messages.extend(history)
        messages.append(HumanMessage(content=user_input))
        
        response = self.model.invoke(messages)
        content = response.content.strip()

        # Attempt to extract JSON first
        json_str = None
        if "```json" in content:
            try:
                json_str = content.split("```json")[1].split("```")[0].strip()
            except IndexError:
                pass
        elif "```" in content:
             try:
                json_str = content.split("```")[1].split("```")[0].strip()
             except IndexError:
                pass
        
        # If no markdown blocks found, check if the content itself looks like JSON
        if not json_str and content.strip().startswith("{") and content.strip().endswith("}"):
            json_str = content.strip()

        parsed_result = None
        if json_str:
            try:
                parsed_result = json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # If we have valid JSON, we can proceed regardless of round count
        if parsed_result:
            parsed_intent = parsed_result.get("intent", "general_qa")
            parsed_task_params = parsed_result.get("task_params", {}) or {}

            if parsed_intent in {"generation", "optimization", "evaluation", "synthesis_planning"}:
                if "run_retrosynthesis" not in parsed_task_params:
                    parsed_task_params["run_retrosynthesis"] = False
                if "skip_questions" not in parsed_task_params:
                    parsed_task_params["skip_questions"] = False
            parsed_result["task_params"] = parsed_task_params

            # Combine summary and analysis for the final response
            summary = parsed_result.get("summary", "Task ready.")
            analysis = parsed_result.get("model_analysis", [])
            if analysis:
                summary += "\n\n**Model Selection Analysis**:\n" + "\n".join([f"- {item}" for item in analysis])

            if parsed_intent in {"generation", "optimization"}:
                retro_enabled = bool(parsed_task_params.get("run_retrosynthesis", False))
                summary += f"\n\nRetrosynthesis: {'enabled' if retro_enabled else 'disabled'}."

            return {
                "intent": parsed_intent,
                "task_params": parsed_task_params,
                "current_agent": "IntentAgent",
                "results": {"response": summary}
            }

        # If NO JSON and count < 1, then we need clarification
        if not skip_questions and clarification_round_count < 1:
            return {
                "intent": "clarification_needed",
                "task_params": task_params,
                "current_agent": "IntentAgent",
                "results": {"response": f"{clarify_marker}\n" + content},
            }
        
        # Fallback logic if count >= 1 but parsing failed (or other cases)
        content_lower = content.lower()
        mentions_retro = ("retrosynthesis" in content_lower) or ("逆合成" in content)
        mentions_eval = ("evaluate" in content_lower) or ("evaluation" in content_lower) or ("评估" in content) or ("评价" in content)
        likely_generation = any(
            key in content_lower
            for key in [
                "design",
                "generate",
                "generation",
                "denovo",
                "de novo",
                "inhibitor",
                "optimiz",
            ]
        ) or any(key in content for key in ["生成", "设计", "优化"])

        # Natural Language Response (Clarification needed)
        return {
            "intent": "clarification_needed",
            "task_params": task_params,
            "current_agent": "IntentAgent",
            "results": {"response": f"{clarify_marker}\n" + content}
        }

# LangGraph Node Wrapper
def intent_agent_node(state: AgentState) -> Dict[str, Any]:
    agent = IntentAgent()
    return agent.run(state)
