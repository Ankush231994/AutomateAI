import os

# RAG/Embeddings/FAISS settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_DIR = os.path.join(BASE_DIR, 'rag')
EMBEDDINGS_DIR = os.path.join(RAG_DIR, 'embeddings')
EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, 'chunk_embeddings.pkl')
FAISS_INDEX_FILE = os.path.join(EMBEDDINGS_DIR, 'faiss.index')

# Model and LLM settings
MODEL_NAME = os.getenv("LLM_MODEL", "deepseek-r1:1.5b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", 60))

# Database settings (example for SQLite)
DB_FILE = os.path.join(BASE_DIR, '..', 'db.sqlite')
DB_URL = f"sqlite:///{os.path.abspath(DB_FILE)}"

# Add other shared/configurable settings here as needed 