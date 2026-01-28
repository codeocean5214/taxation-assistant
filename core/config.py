"""
Configuration for RAG system - LangChain v1.2+
Updated for latest best practices (January 2025)
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
VECTOR_DB_DIR = DATA_DIR / "vector_db"

# Ensure directories exist
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

# Document processing settings
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap to preserve context

# Embedding model (HuggingFace - free, runs locally)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Ollama LLM settings
# Make sure you've run: ollama pull llama3.2
OLLAMA_MODEL = "llama3.2"
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_TEMPERATURE = 0.1  # Low for factual responses
LLM_NUM_PREDICT = 512  # Max tokens to generate

# Retrieval settings
TOP_K_RETRIEVAL = 4  # Number of relevant chunks to retrieve

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000

# Streamlit settings
STREAMLIT_PAGE_TITLE = "Tax Policy Assistant"
STREAMLIT_PAGE_ICON = "📊"