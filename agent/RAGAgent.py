import os
import time
from typing import Any, Dict, List, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from agent.base_agent import BaseAgent
from agent.state import AgentState

class RAGAgent(BaseAgent):
    def __init__(self, agent_name: str = "RAGAgent"):
        super().__init__(agent_name)
        self.embeddings = self._init_embeddings()
        self.vector_store = None
        self.docs_paths = [
            "services/API_REFERENCE.md",
            "drugtoolkg/agent_kg.json"
        ]
        # Define persistence path inside agent folder
        self.index_path = os.path.join(os.path.dirname(__file__), "vector_store")
        self._initialize_knowledge_base()

    def _init_embeddings(self) -> OpenAIEmbeddings:
        """Initialize embeddings using the same credentials as the LLM."""
        # Extract credentials from the initialized LLM in BaseAgent
        api_key = None
        # Try to get API key from the model instance
        if hasattr(self.model, "openai_api_key"):
            val = self.model.openai_api_key
            if hasattr(val, "get_secret_value"):
                api_key = val.get_secret_value()
            else:
                api_key = val
        
        # Fallback to environment variables
        if not api_key or api_key == "missing_api_key":
            api_key = os.getenv("RAG_AGENT_LLM_API_KEY") or os.getenv("GLOBAL_LLM_API_KEY")

        base_url = getattr(self.model, "openai_api_base", None) or os.getenv("OPENAI_API_BASE") or os.getenv("GLOBAL_LLM_API_BASE")
        
        if not api_key:
             print("RAGAgent: Warning - No API Key found for Embeddings.")
             api_key = "missing_key"

        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=api_key,
            base_url=base_url
        )

    def _initialize_knowledge_base(self):
        """Load documents, split them, and create/load a vector store."""
        # Check if index exists
        if os.path.exists(os.path.join(self.index_path, "index.faiss")):
            try:
                self.vector_store = FAISS.load_local(self.index_path, self.embeddings, allow_dangerous_deserialization=True)
                print(f"RAGAgent: Loaded existing vector store from {self.index_path}")
                return
            except Exception as e:
                print(f"RAGAgent: Error loading existing index: {e}. Rebuilding...")
        
        # If not found or error, build it
        self.build_index()

    def build_index(self):
        """Explicitly builds the vector index from documents."""
        print("RAGAgent: Building vector index...")
        documents = []
        
        # Resolve paths relative to project root if needed
        project_root = os.path.dirname(os.path.dirname(__file__))
        
        for path in self.docs_paths:
            # Try absolute or relative to project root
            full_path = path if os.path.isabs(path) else os.path.join(project_root, path)
            
            if os.path.exists(full_path):
                try:
                    loader = TextLoader(full_path, encoding='utf-8')
                    documents.extend(loader.load())
                    print(f"RAGAgent: Loaded {full_path}")
                except Exception as e:
                    print(f"RAGAgent: Error loading {full_path}: {e}")
            else:
                print(f"RAGAgent: File not found: {full_path}")

        if not documents:
            print("RAGAgent: No documents loaded.")
            return

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(documents)
        
        if splits:
            try:
                self.vector_store = FAISS.from_documents(splits, self.embeddings)
                # Save to disk
                if not os.path.exists(self.index_path):
                    os.makedirs(self.index_path)
                self.vector_store.save_local(self.index_path)
                print(f"RAGAgent: Initialized and saved vector store with {len(splits)} chunks to {self.index_path}")
            except Exception as e:
                print(f"RAGAgent: Error initializing vector store: {e}")
        else:
            print("RAGAgent: No splits created from documents.")

    def retrieve(self, query: str, k: int = 3) -> str:
        """Retrieve relevant context for a given query."""
        if not self.vector_store:
            return "Knowledge base not initialized."
        
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            return "\n\n".join([f"Source: {d.metadata.get('source', 'unknown')}\nContent:\n{d.page_content}" for d in docs])
        except Exception as e:
            return f"Error during retrieval: {e}"

    def run(self, state: AgentState) -> Dict[str, Any]:
        """
        Process the current state to retrieve relevant information.
        Typically called when 'intent' is unclear or needs more context.
        """
        user_input = state.get("user_input", "")
        current_task = state.get("task_params", {}).get("description", user_input)
        
        if current_task:
            context = self.retrieve(current_task)
            # Store the retrieved context in the results
            return {
                "results": {
                    **state.get("results", {}),
                    "rag_context": context
                }
            }
        return {}

    def add_experience(self, task: str, result_summary: str, logs: str):
        """Adds a completed task experience to the vector store."""
        if not self.vector_store:
            # Try to build or load
            self._initialize_knowledge_base()
            
        if not self.vector_store:
             print("RAGAgent: Failed to initialize vector store. Cannot save experience.")
             return

        # Create a document representing this experience
        # Truncate logs to avoid token limit (approx 8k tokens for embedding model)
        if len(logs) > 20000:
            logs = logs[:20000] + "...(truncated)"
            
        content = f"Task: {task}\n\nSummary: {result_summary}\n\nExecution Logs:\n{logs}"
        metadata = {"source": "execution_history", "type": "experience", "timestamp": str(time.time())}
        
        doc = Document(page_content=content, metadata=metadata)
        
        try:
            # Add to vector store
            self.vector_store.add_documents([doc])
            # Save to disk
            self.vector_store.save_local(self.index_path)
            print("RAGAgent: Experience saved to knowledge base.")
        except Exception as e:
            print(f"RAGAgent: Error saving experience: {e}")
