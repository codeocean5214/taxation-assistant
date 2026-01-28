"""
Document Processor - LangChain v1.2+
Handles PDF loading and intelligent text chunking
"""

from typing import List
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.config import CHUNK_SIZE, CHUNK_OVERLAP


class DocumentProcessor:
    """Processes PDFs into semantically coherent chunks."""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        """
        Initialize document processor.
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # RecursiveCharacterTextSplitter - splits intelligently
        # Priority: paragraphs → sentences → words
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            is_separator_regex=False
        )
    
    def load_pdf(self, pdf_path: str) -> List[Document]:
        """
        Load PDF and extract text with metadata.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of Document objects (one per page)
        """
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        print(f"📄 Loading PDF: {pdf_file.name}")
        
        loader = PyPDFLoader(str(pdf_file))
        documents = loader.load()
        
        print(f"✅ Loaded {len(documents)} pages")
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller, overlapping chunks.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of chunked Documents with preserved metadata
        """
        print(f"✂️  Chunking {len(documents)} documents...")
        
        chunks = self.text_splitter.split_documents(documents)
        
        print(f"✅ Created {len(chunks)} chunks")
        return chunks
    
    def process_pdf(self, pdf_path: str) -> List[Document]:
        """
        Complete pipeline: load → chunk → return.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of ready-to-embed Document chunks
        """
        documents = self.load_pdf(pdf_path)
        chunks = self.chunk_documents(documents)
        
        if chunks:
            print(f"\n📝 Sample chunk:")
            print(f"Content: {chunks[0].page_content[:200]}...")
            print(f"Metadata: {chunks[0].metadata}\n")
        
        return chunks