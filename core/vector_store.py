"""
Vector Store - LangChain v1.2+
Handles embedding generation and similarity search using FAISS
"""

from typing import List, Optional
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from core.config import EMBEDDING_MODEL_NAME, VECTOR_DB_DIR, TOP_K_RETRIEVAL


class VectorStore:
    """Manages document embeddings and semantic search."""
    
    def __init__(self, embedding_model_name: str = EMBEDDING_MODEL_NAME):
        """
        Initialize vector store with embedding model.
        
        Args:
            embedding_model_name: HuggingFace model name
        """
        print(f"🔧 Initializing embeddings: {embedding_model_name}")
        
        # HuggingFace embeddings - runs locally, no API key needed
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'},  # Use 'cuda' if GPU available
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.vector_store: Optional[FAISS] = None
        self.db_path = VECTOR_DB_DIR
    
    def create_vector_store(self, chunks: List[Document]) -> FAISS:
        """
        Create FAISS index from document chunks.
        
        Args:
            chunks: List of Document objects to embed
            
        Returns:
            FAISS vector store
        """
        if not chunks:
            raise ValueError("Cannot create vector store from empty chunks")
        
        print(f"🔢 Creating embeddings for {len(chunks)} chunks...")
        
        # Create FAISS index with cosine similarity
        self.vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        
        print(f"✅ Vector store created")
        return self.vector_store
    
    def save_vector_store(self, name: str = "tax_policy_index"):
        """
        Persist vector store to disk.
        
        Args:
            name: Index name
        """
        if self.vector_store is None:
            raise ValueError("No vector store to save")
        
        save_path = self.db_path / name
        self.vector_store.save_local(str(save_path))
        print(f"💾 Saved to: {save_path}")
    
    def load_vector_store(self, name: str = "tax_policy_index") -> FAISS:
        """
        Load persisted vector store.
        
        Args:
            name: Index name
            
        Returns:
            Loaded FAISS vector store
        """
        load_path = self.db_path / name
        
        if not load_path.exists():
            raise FileNotFoundError(f"Vector store not found: {load_path}")
        
        print(f"📂 Loading vector store from: {load_path}")
        
        self.vector_store = FAISS.load_local(
            str(load_path),
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        print("✅ Vector store loaded")
        return self.vector_store
    
    def similarity_search(self, query: str, k: int = TOP_K_RETRIEVAL) -> List[Document]:
        """
        Find most relevant chunks for a query.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of most similar Documents
        """
        if self.vector_store is None:
            raise ValueError("No vector store available")
        
        return self.vector_store.similarity_search(query, k=k)
    
    def as_retriever(self, k: int = TOP_K_RETRIEVAL):
        """
        Get retriever interface (for use with chains).
        
        Args:
            k: Number of documents to retrieve
            
        Returns:
            Retriever object
        """
        if self.vector_store is None:
            raise ValueError("No vector store available")
        
        return self.vector_store.as_retriever(search_kwargs={"k": k})


def build_vector_store(chunks: List[Document], name: str = "tax_policy_index") -> VectorStore:
    """
    Build and save vector store from chunks.
    
    Args:
        chunks: Document chunks
        name: Index name
        
    Returns:
        VectorStore instance
    """
    store = VectorStore()
    store.create_vector_store(chunks)
    store.save_vector_store(name)
    return store