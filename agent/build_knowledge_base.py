import os
import sys
from dotenv import load_dotenv

# Add project root to path so we can import agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from agent/.env
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(env_path)

from agent.RAGAgent import RAGAgent

def main():
    print("Starting Knowledge Base Build Process...")
    
    # Initialize agent (this will trigger _initialize_knowledge_base which calls build_index if missing)
    # But to force a rebuild, we call build_index explicitly
    agent = RAGAgent()
    
    print("\n--- Forcing Rebuild of Index ---")
    agent.build_index()
    
    print("\nBuild Complete.")
    print(f"Index saved to: {agent.index_path}")

if __name__ == "__main__":
    main()
