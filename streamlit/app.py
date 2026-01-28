"""
Streamlit App - LangChain v1.2+
Interactive Tax Policy Assistant
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from core.document_processor import DocumentProcessor
from core.vector_store import build_vector_store
from core.rag_engine import RAGEngine
from core.vector_store import VectorStore
from core.config import (
    STREAMLIT_PAGE_TITLE,
    STREAMLIT_PAGE_ICON,
    DOCUMENTS_DIR,
    OLLAMA_MODEL
)

# Page config
st.set_page_config(
    page_title=STREAMLIT_PAGE_TITLE,
    page_icon=STREAMLIT_PAGE_ICON,
    layout="wide"
)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None
if "vector_store_ready" not in st.session_state:
    st.session_state.vector_store_ready = False


def initialize_rag():
    """Initialize RAG engine."""
    try:
        vector_store = VectorStore()
        vector_store.load_vector_store("tax_policy_index")
        st.session_state.rag_engine = RAGEngine(vector_store)
        st.session_state.vector_store_ready = True
        return True
    except FileNotFoundError:
        st.session_state.vector_store_ready = False
        return False
    except Exception as e:
        st.error(f"Error: {e}")
        return False


def process_pdf(uploaded_file):
    """Process uploaded PDF."""
    with st.spinner("📄 Processing PDF..."):
        # Save file
        pdf_path = DOCUMENTS_DIR / uploaded_file.name
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process
        processor = DocumentProcessor()
        chunks = processor.process_pdf(str(pdf_path))
        
        # Build vector store
        with st.spinner("🔢 Building vector database..."):
            build_vector_store(chunks, "tax_policy_index")
        
        st.success(f"✅ Processed {len(chunks)} chunks")
        initialize_rag()


# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    
    st.markdown("### 📁 Document Management")
    
    if not st.session_state.vector_store_ready:
        initialize_rag()
    
    if st.session_state.vector_store_ready:
        st.success("✅ Vector store loaded")
        st.info(f"🤖 Model: {OLLAMA_MODEL}")
    else:
        st.warning("⚠️ Upload a PDF to begin")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Tax Policy PDF",
        type=["pdf"]
    )
    
    if uploaded_file and st.button("Process PDF"):
        process_pdf(uploaded_file)
        st.rerun()
    
    st.markdown("---")
    
    # Settings
    with st.expander("🔧 Advanced"):
        k = st.slider("Context chunks", 1, 10, 4)
        st.session_state.k = k
    
    st.markdown("---")
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown(f"""
    ### ℹ️ About
    
    **Tech Stack:**
    - LangChain v1.2+
    - Llama 3.2 (via Ollama)
    - FAISS vector store
    - HuggingFace embeddings
    
    **Prerequisites:**
    ```bash
    ollama pull llama3.2
    ollama serve
    ```
    """)

# Main interface
st.title(f"{STREAMLIT_PAGE_ICON} Tax Policy Assistant")
st.markdown("Ask questions about corporate taxation policies based on your documents.")

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 Sources"):
                for i, src in enumerate(msg["sources"], 1):
                    page = src['metadata'].get('page', 'N/A')
                    st.markdown(f"**Source {i}** (Page {page})")
                    st.text(src["content"][:300] + "...")
                    st.markdown("---")

# Chat input
if prompt := st.chat_input("Ask about tax policy..."):
    if not st.session_state.vector_store_ready:
        st.error("⚠️ Please upload and process a PDF first")
        st.stop()
    
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                k = st.session_state.get("k", 4)
                response = st.session_state.rag_engine.query(
                    prompt,
                    return_sources=True,
                    k=k
                )
                
                answer = response["answer"]
                sources = response.get("sources", [])
                
                st.markdown(answer)
                
                if sources:
                    with st.expander("📚 Sources"):
                        for i, src in enumerate(sources, 1):
                            page = src['metadata'].get('page', 'N/A')
                            st.markdown(f"**Source {i}** (Page {page})")
                            st.text(src["content"][:300] + "...")
                            st.markdown("---")
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.info("Ensure Ollama is running: `ollama serve`")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    💡 Powered by LangChain v1.2+ | 🔒 100% local processing
</div>
""", unsafe_allow_html=True)