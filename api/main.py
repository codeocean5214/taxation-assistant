"""
FastAPI Backend - LangChain v1.2+
REST API for Tax Policy RAG system
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from core.vector_store import VectorStore
from core.rag_engine import RAGEngine
from core.config import OLLAMA_MODEL, API_HOST, API_PORT


# Request/Response models
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    return_sources: bool = True
    k: int = Field(4, ge=1, le=10)


class BatchQueryRequest(BaseModel):
    questions: List[str] = Field(..., min_items=1, max_items=10)


class QueryResponse(BaseModel):
    answer: str
    sources: Optional[List[Dict[str, Any]]] = None


class BatchQueryResponse(BaseModel):
    results: List[QueryResponse]


# Initialize FastAPI
app = FastAPI(
    title="Tax Policy RAG API",
    description="LangChain v1.2 + Llama 3.2 RAG API",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG engine
rag_engine: Optional[RAGEngine] = None


@app.on_event("startup")
async def startup_event():
    """Initialize RAG engine on startup."""
    global rag_engine
    try:
        print("🚀 Starting Tax Policy RAG API...")
        
        vector_store = VectorStore()
        vector_store.load_vector_store("tax_policy_index")
        rag_engine = RAGEngine(vector_store)
        
        print(f"✅ RAG engine initialized with {OLLAMA_MODEL}")
    except FileNotFoundError:
        print("⚠️ Vector store not found. Run: python scripts/build_index.py")
        rag_engine = None
    except Exception as e:
        print(f"❌ Error: {e}")
        rag_engine = None


@app.get("/")
async def root():
    """API info."""
    return {
        "name": "Tax Policy RAG API",
        "version": "1.0.0",
        "langchain_version": "1.2+",
        "model": OLLAMA_MODEL,
        "status": "ready" if rag_engine else "not_initialized"
    }


@app.get("/health")
async def health():
    """Health check."""
    if rag_engine is None:
        raise HTTPException(503, "RAG engine not initialized")
    
    return {"status": "healthy", "model": OLLAMA_MODEL}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Process a single query.
    
    Example:
        POST /query
        {
            "question": "What is the corporate tax rate?",
            "return_sources": true,
            "k": 4
        }
    """
    if rag_engine is None:
        raise HTTPException(503, "RAG engine not initialized")
    
    try:
        result = rag_engine.query(
            question=request.question,
            return_sources=request.return_sources,
            k=request.k
        )
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(500, f"Query failed: {str(e)}")


@app.post("/batch_query", response_model=BatchQueryResponse)
async def batch_query(request: BatchQueryRequest):
    """Process multiple queries."""
    if rag_engine is None:
        raise HTTPException(503, "RAG engine not initialized")
    
    try:
        results = rag_engine.batch_query(request.questions)
        return BatchQueryResponse(
            results=[QueryResponse(**r) for r in results]
        )
    except Exception as e:
        raise HTTPException(500, f"Batch query failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)