import streamlit as st
import os
import sys
import time
import datetime
import uuid
import json
import pandas as pd
import io
from dotenv import load_dotenv

import re

# --- Stream Logger for Real-time Stdout Capture and File Logging ---
class StreamLogger:
    def __init__(self, original_stdout, status_container, log_file=None):
        self.original_stdout = original_stdout
        self.status_container = status_container
        self.agent_logs = {}  # { 'AgentName': [lines] }
        self.current_agent = "System"
        self.log_file = log_file
        
        # Open log file if provided
        if self.log_file:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            self.file_handle = open(self.log_file, "a", encoding="utf-8")
        else:
            self.file_handle = None
        
    def write(self, message):
        # 1. Write to original stdout (console)
        self.original_stdout.write(message)
        # self.original_stdout.flush() # Optional, can be noisy
        
        # 2. Write to log file
        if self.file_handle:
            self.file_handle.write(message)
            self.file_handle.flush()
        
        # 3. Filter and write to Streamlit status
        if message.strip():
            clean_msg = message.strip()
            
            # Detect Agent Tag: [AgentName]
            # Regex looks for start of line [Name] or **[Name]
            match = re.match(r'^(\*\*)?\[(\w+)\]', clean_msg)
            if match:
                # Group 2 is the agent name because Group 1 is optional '**'
                self.current_agent = match.group(2)
                
            # Store in structured dict
            if self.current_agent not in self.agent_logs:
                self.agent_logs[self.current_agent] = []
            self.agent_logs[self.current_agent].append(clean_msg)
            
            # Use st.markdown to display log lines inside the status container
            # Bold the line if it starts with [AgentName] to make it stand out in real-time
            with self.status_container:
                if match:
                    st.markdown(f"**{clean_msg}**")
                else:
                    st.text(clean_msg)
            
    def flush(self):
        self.original_stdout.flush()

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import the workflow app and state definition
from agent.interactive_workflow import app, AgentState
from agent.RAGAgent import RAGAgent
from langchain_core.messages import HumanMessage, AIMessage
from streamlit_pdf_viewer import pdf_viewer

# Load environment variables
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="Drug Discovery Agent",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for "Beautiful" Look ---
st.markdown("""
<style>
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stButton button {
        border-radius: 20px;
    }
    h1 {
        color: #2e86c1;
    }
    .report-box {
        border: 2px solid #28b463;
        padding: 15px;
        border-radius: 10px;
        background-color: #eafaf1;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your AI Drug Discovery Agent. I can help you with target preparation, molecule generation, docking evaluation, and synthesis planning.\n\nPlease upload any necessary data (PDB, SMILES) in the sidebar, or just describe your task!"}
    ]

if "awaiting_intent_clarification" not in st.session_state:
    st.session_state.awaiting_intent_clarification = False

if "intent_turn_id" not in st.session_state:
    st.session_state.intent_turn_id = None

if "run_id" not in st.session_state:
    st.session_state.run_id = None

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}

# --- Sidebar ---
with st.sidebar:
    st.title("🛠️ Control Panel")
    st.markdown("### Data Upload")
    
    uploaded_pdb = st.file_uploader("Upload Target PDB (.pdb)", type="pdb")
    if uploaded_pdb:
        # Save file
        save_dir = os.path.join("data", "uploads")
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, uploaded_pdb.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_pdb.getbuffer())
        st.session_state.uploaded_files["pdb_path"] = file_path
        st.success(f"Loaded PDB: {uploaded_pdb.name}")

    uploaded_csv = st.file_uploader("Upload Molecules (.csv)", type="csv")
    if uploaded_csv:
        # Save file
        save_dir = os.path.join("data", "uploads")
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, uploaded_csv.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_csv.getbuffer())
        st.session_state.uploaded_files["csv_path"] = file_path
        st.success(f"Loaded CSV: {uploaded_csv.name}")

    st.markdown("---")
    st.markdown("### System Status")
    if st.session_state.run_id:
        st.info(f"Run ID: {st.session_state.run_id}")
    else:
        st.write("Ready to start.")

# --- Main Interface ---
st.title("💊 AI Drug Discovery Assistant")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# Chat Input
if prompt := st.chat_input("Describe your drug discovery task..."):
    # 1. Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt, unsafe_allow_html=True)

    # 2. Prepare Agent Execution
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        status_container = st.status("🤖 Agents working...", expanded=True)
        
        is_followup_answer = bool(st.session_state.awaiting_intent_clarification and st.session_state.intent_turn_id)
        if not is_followup_answer:
            st.session_state.run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
            st.session_state.intent_turn_id = str(uuid.uuid4())[:8]
        
        # Initialize State
        current_task_params = st.session_state.uploaded_files.copy()
        if st.session_state.intent_turn_id:
            current_task_params["intent_turn_id"] = st.session_state.intent_turn_id
        
        # Convert history to LangChain format for Agent Memory
        history_langchain = []
        # We iterate over all messages EXCEPT the last one (which is the current user prompt)
        # The current prompt is passed as 'user_input' and handled by the agent.
        for msg in st.session_state.messages[:-1]:
            if msg["role"] == "user":
                history_langchain.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                history_langchain.append(AIMessage(content=msg["content"]))

        initial_state = AgentState(
            messages=history_langchain, 
            user_input=prompt,
            intent=None,
            task_params=current_task_params,
            current_agent="IntentAgent",
            results={},
            error=None,
            is_complete=False,
            run_id=st.session_state.run_id,
            loop_count=0
        )

        # 3. Stream/Run Graph
        # We use stream to get updates from each node
        full_response = ""
        final_state = None
        previous_logs_len = 0
        
        # --- Capture Stdout ---
        original_stdout = sys.stdout
        
        # Define log file
        log_dir = os.path.join("logs")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Use run_id if available, otherwise just timestamp
        rid = st.session_state.run_id if st.session_state.run_id else "init"
        log_file = os.path.join(log_dir, f"streamlit_run_{timestamp}_{rid}.log")
        
        stream_logger = StreamLogger(original_stdout, status_container, log_file=log_file)
        sys.stdout = stream_logger
        print(f"[System] Streamlit logging started. File: {log_file}")
        
        try:
            for step in app.stream(initial_state):
                # 'step' is a dict where key is node name and value is the state update
                for node_name, state_update in step.items():
                    # Update status - Simplified notification since logs are streaming
                    # status_container.write(f"✅ **{node_name}** completed step.")
                    
                    # Extract logs/updates to show to user - DEPRECATED
                    # Since we are streaming stdout, we don't need to manually parse agent_logs from state anymore
                    # This avoids duplication and delay
                    
                    # Check for intermediate results
                    if node_name == "intent_agent":
                        intent = state_update.get("intent")
                        # status_container.markdown(f"**Intent Detected**: `{intent}`")
                    
                    elif node_name == "generator_agent":
                        gen_res = state_update.get("results", {}).get("generation", {})
                        # Calculate total count from all tools
                        count = 0
                        for tool_res in gen_res.values():
                            if isinstance(tool_res, dict):
                                if "molecules" in tool_res:
                                    count += len(tool_res["molecules"])
                                elif "count" in tool_res:
                                    count += tool_res["count"]
                                elif "smiles" in tool_res:
                                    count += len(tool_res["smiles"])
                        
                        status_container.markdown(f"⚗️ **Generated**: {count} molecules")
                        
                    elif node_name == "evaluator_agent":
                        eval_res = state_update.get("results", {}).get("evaluation", {})
                        qualified = eval_res.get("qualified_count", 0)
                        status_container.markdown(f"⚖️ **Qualified**: {qualified} candidates")
                    
                    elif node_name == "report_agent":
                        status_container.markdown("📄 **Report Generated**")

                final_state = state_update # Keep updating final state

            status_container.update(label="✅ Task Completed!", state="complete", expanded=False)
            
        except Exception as e:
            status_container.update(label="❌ Error Occurred", state="error")
            st.error(f"An error occurred: {e}")
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})
        
        finally:
            # Restore stdout
            sys.stdout = original_stdout

        # 4. Final Response Construction
        
        # Only check for report if the task is finished
        if final_state and final_state.get("is_complete", False):
            # Define the fixed report directory
            # ReportAgent saves to data/reports/{timestamp}
            # We search for the MOST RECENT subdirectory in data/reports
            base_report_dir = os.path.join("data", "reports")
            
            latest_report_dir = None
            if os.path.exists(base_report_dir):
                all_subdirs = [os.path.join(base_report_dir, d) for d in os.listdir(base_report_dir) if os.path.isdir(os.path.join(base_report_dir, d))]
                if all_subdirs:
                    latest_report_dir = max(all_subdirs, key=os.path.getmtime)
            
            if latest_report_dir:
                # Look for HTML, PDF, or MD
                report_files = [f for f in os.listdir(latest_report_dir) if f.endswith('.html') or f.endswith('.pdf') or f.endswith('.md')]
                
                if report_files:
                    full_response += "### 📄 Report Ready\n"
                    for rf in report_files:
                        file_path = os.path.join(latest_report_dir, rf)
                        
                        # 1. Download Button
                        with open(file_path, "rb") as f:
                            file_data = f.read()
                            st.download_button(
                                label=f"Download {rf}",
                                data=file_data,
                                file_name=rf,
                                mime="application/pdf" if rf.endswith(".pdf") else ("text/html" if rf.endswith(".html") else "text/markdown"),
                                key=f"btn_{rf}_{uuid.uuid4()}" # Unique key
                            )
                        
                        # 2. PDF Preview
                        if rf.endswith(".pdf"):
                            full_response += f"\n**Previewing {rf}:**\n"
                            with message_placeholder.container():
                                 pdf_viewer(file_path, width=700, height=800)
        
        # Add textual summary
        if final_state:
            # Use the last message from the graph if available, or construct one
            if final_state.get("messages") and len(final_state["messages"]) > 0:
                # The last message from IntentAgent might be too early if other agents ran.
                # Let's check results.
                pass
            
            # Construct summary from logs
            logs = final_state.get("task_params", {}).get("agent_logs", "")
            if logs:
                full_response += "\n\n**Key Results:**\n" + logs

            if final_state.get("intent") == "clarification_needed":
                st.session_state.awaiting_intent_clarification = True
            else:
                st.session_state.awaiting_intent_clarification = False
                st.session_state.intent_turn_id = None
            
        # Add full raw logs (Traceability) - Split by Agent
        if stream_logger.agent_logs:
            full_response += "\n\n### 🕵️ Agent Workflow Trace\n"
            for agent_name, logs in stream_logger.agent_logs.items():
                # Skip System logs if empty or trivial
                if agent_name == "System" and not logs:
                    continue
                    
                log_content = "\n".join(logs)
                # Create a collapsible section for each agent
                full_response += f"\n<details><summary><b>[{agent_name}]</b> (Click to expand)</summary>\n\n```text\n{log_content}\n```\n</details>"
            
        if not full_response.strip():
            full_response = "Task executed."

        message_placeholder.markdown(full_response, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
