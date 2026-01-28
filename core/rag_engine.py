"""
RAG Engine - LangChain v1.2+
Uses ChatOllama with Llama 3.2 for question answering
"""

from typing import List, Dict, Any, TYPE_CHECKING
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from core.config import (
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    LLM_TEMPERATURE,
    LLM_NUM_PREDICT,
    TOP_K_RETRIEVAL
)

if TYPE_CHECKING:
    from core.vector_store import VectorStore


class RAGEngine:
    """
    RAG (Retrieval-Augmented Generation) Engine.
    LangChain v1.2 implementation with ChatOllama.
    """
    
    def __init__(
        self,
        vector_store: "VectorStore",
        model_name: str = OLLAMA_MODEL,
        temperature: float = LLM_TEMPERATURE
    ):
        """
        Initialize RAG engine.
        
        Args:
            vector_store: Initialized VectorStore
            model_name: Ollama model name (e.g., 'llama3.2')
            temperature: LLM temperature (0-1)
        """
        self.vector_store = vector_store
        
        print(f"🤖 Initializing ChatOllama: {model_name}")
        
        # ChatOllama - LangChain v1 official integration
        self.llm = ChatOllama(
            model=model_name,
            base_url=OLLAMA_BASE_URL,
            temperature=temperature,
            num_predict=LLM_NUM_PREDICT
        )
        
        # Create RAG chain
        self.chain = self._create_rag_chain()
        
        print("✅ RAG Engine initialized")
    
    def _create_rag_chain(self):
        """
        Create RAG chain using LangChain v1 LCEL (LangChain Expression Language).
        
        Returns:
            Runnable chain
        """
        # Prompt template for tax policy Q&A
        template = """You are an expert assistant for corporate taxation policy.
Use the following context to answer the question. If you cannot answer based on the context, say so.

Context:
{context}

Question: {question}

Provide a clear, evidence-based answer:"""

        prompt = ChatPromptTemplate.from_template(template)
        
        # Create chain using LCEL
        # retriever | format_docs | prompt | llm | parse
        def format_docs(docs: List[Document]) -> str:
            return "\n\n---\n\n".join([d.page_content for d in docs])
        
        chain = (
            {
                "context": self.vector_store.as_retriever() | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    def query(
        self,
        question: str,
        return_sources: bool = True,
        k: int = TOP_K_RETRIEVAL
    ) -> Dict[str, Any]:
        """
        Ask a question and get an answer with sources.
        
        Args:
            question: User's question
            return_sources: Include source documents
            k: Number of source chunks to retrieve
            
        Returns:
            Dict with 'answer' and optionally 'sources'
        """
        print(f"\n🔍 Query: {question[:100]}...")
        
        # Retrieve relevant documents
        relevant_docs = self.vector_store.similarity_search(question, k=k)
        
        if not relevant_docs:
            return {
                "answer": "No relevant information found in the documents.",
                "sources": []
            }
        
        print(f"📚 Retrieved {len(relevant_docs)} chunks")
        
        # Generate answer using chain
        print("💭 Generating answer...")
        answer = self.chain.invoke(question)
        
        response = {"answer": answer.strip()}
        
        if return_sources:
            response["sources"] = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in relevant_docs
            ]
        
        print("✅ Answer generated\n")
        return response
    
    def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple questions.
        
        Args:
            questions: List of questions
            
        Returns:
            List of answer dictionaries
        """
        return [self.query(q, return_sources=False) for q in questions]


def initialize_rag_engine(vector_store_name: str = "tax_policy_index") -> RAGEngine:
    """
    Load vector store and create RAG engine.
    
    Args:
        vector_store_name: Name of saved vector store
        
    Returns:
        Initialized RAG engine
    """
    from core.vector_store import VectorStore
    
    vector_store = VectorStore()
    vector_store.load_vector_store(vector_store_name)
    
    return RAGEngine(vector_store)