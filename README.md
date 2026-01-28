# Tax Policy RAG System 📊

**LangChain v1.2+ | Llama 3.2 | 100% Free & Local**

A production-ready RAG (Retrieval-Augmented Generation) system for corporate taxation policy Q&A, built with the latest LangChain v1.2 stack.

## ✨ Features

- 🆕 **LangChain v1.2+**: Latest streamlined API
- 🤖 **Llama 3.2**: Via Ollama (free, local)
- 📄 **PDF Processing**: Intelligent chunking
- 🔍 **Semantic Search**: FAISS vector store
- 💬 **Chat UI**: Streamlit interface
- 🌐 **REST API**: FastAPI backend
- 🔒 **Privacy**: 100% local processing

## 🏗️ Architecture

```
┌──────────────┐     ┌──────────────┐
│  Streamlit   │────▶│              │
│     UI       │     │  RAG Engine  │
└──────────────┘     │  (LangChain  │
                     │     v1.2)    │
┌──────────────┐     │              │
│   FastAPI    │────▶│              │
│     API      │     └──────────────┘
└──────────────┘            │
                   ┌────────┴────────┐
                   ▼                 ▼
            ┌──────────┐      ┌──────────┐
            │  FAISS   │      │ChatOllama│
            │  Vector  │      │(Llama3.2)│
            │  Store   │      └──────────┘
            └──────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** (LangChain v1 requires 3.10+)
- **Ollama** installed and running

### 1. Install Ollama & Llama 3.2

```bash
# Install Ollama
# Windows/Mac: Download from https://ollama.com
# Linux:
curl -fsSL https://ollama.com/install.sh | sh

# Pull Llama 3.2
ollama pull llama3.2

# Start Ollama server (keep running)
ollama serve
```

### 2. Setup Python Environment

```bash
# Clone or navigate to project
cd tax-policy-rag

# Create virtual environment
python -m venv venv

# Activate
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install --prefer-binary -r requirements.txt
```

### 3. Process Your PDF

```bash
# Place PDF in data/documents/
# Then build the index:
python scripts/build_index.py data/documents/your_tax_policy.pdf
```

### 4. Run the Application

**Option A: Streamlit UI**
```bash
streamlit run streamlit_app/app.py
```
Open http://localhost:8501

**Option B: FastAPI**
```bash
uvicorn api.main:app --reload
```
API docs at http://localhost:8000/docs

**Option C: CLI**
```bash
python scripts/test_query.py
```

## 📁 Project Structure

```
tax-policy-rag/
├── core/                   # Shared RAG logic
│   ├── config.py          # Configuration
│   ├── document_processor.py  # PDF processing
│   ├── vector_store.py    # FAISS embeddings
│   └── rag_engine.py      # Query processing
├── api/
│   └── main.py            # FastAPI endpoints
├── streamlit_app/
│   └── app.py             # Streamlit UI
├── scripts/
│   ├── build_index.py     # Index builder
│   └── test_query.py      # CLI tester
├── data/
│   ├── documents/         # PDFs
│   └── vector_db/         # FAISS index
└── requirements.txt
```

## 🔧 Configuration

Edit `core/config.py`:

```python
# Model selection
OLLAMA_MODEL = "llama3.2"  # or "mistral", "phi3", etc.

# Chunking strategy
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval
TOP_K_RETRIEVAL = 4  # Chunks per query
```

## 💡 Usage Examples

### Streamlit
1. Upload PDF via sidebar
2. Click "Process PDF"
3. Ask questions in chat

### API

**Single Query:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the corporate tax rate?",
    "return_sources": true,
    "k": 4
  }'
```

**Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/query",
    json={"question": "Explain R&D tax credits"}
)
print(response.json()["answer"])
```

## 🆕 What's New in LangChain v1.2

This project uses LangChain v1.2+ features:

- ✅ **ChatOllama**: Official Ollama integration via `langchain-ollama`
- ✅ **LCEL**: LangChain Expression Language for chains
- ✅ **Streamlined imports**: Clean v1 namespace
- ✅ **Better type hints**: Full typing support
- ✅ **Improved docs**: Unified documentation

### Migration from v0.x

Key changes:
```python
# Old (LangChain 0.x)
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

# New (LangChain v1.2+)
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
```

## 🐛 Troubleshooting

**Ollama not found:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not, start it:
ollama serve
```

**Model not found:**
```bash
# List installed models
ollama list

# Install llama3.2
ollama pull llama3.2
```

**Import errors:**
```bash
# Clean reinstall
pip uninstall langchain langchain-community langchain-core langchain-ollama -y
pip install --prefer-binary -r requirements.txt
```

**Slow responses:**
- First query loads model (~5-10 sec)
- Subsequent queries: 3-5 sec
- Use smaller model for faster: `ollama pull phi3`

## 📚 Resources

- [LangChain v1 Docs](https://docs.langchain.com)
- [Ollama Models](https://ollama.com/library)
- [FAISS Documentation](https://github.com/facebookresearch/faiss/wiki)
- [LangChain v1 Migration Guide](https://docs.langchain.com/oss/python/releases/langchain-v1)

## 🎯 Performance

- **First query**: 10-15 sec (model loading)
- **Subsequent**: 3-5 sec
- **RAM**: 4-6GB (for Llama 3.2)
- **Disk**: 2-3GB (model + vectors)

## 🤝 Contributing

This is a learning project demonstrating LangChain v1.2 best practices. Feel free to:
- Experiment with different models
- Improve prompts
- Add evaluation metrics
- Build multi-document support

## 📄 License

MIT License

## 🙏 Acknowledgments

- **LangChain**: Modern LLM framework
- **Ollama**: Local LLM deployment
- **Meta**: Llama 3.2 model
- **FAISS**: Fast similarity search

---

**Built with LangChain v1.2+ | January 2025**
