"""
Build Vector Index - LangChain v1.2+
Process PDF and create FAISS vector database

Usage:
    python scripts/build_index.py path/to/document.pdf
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from core.document_processor import DocumentProcessor
from core.vector_store import build_vector_store


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/build_index.py <pdf_path>")
        print("\nExample:")
        print("  python scripts/build_index.py data/documents/tax_policy.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    if not Path(pdf_path).exists():
        print(f"❌ PDF not found: {pdf_path}")
        sys.exit(1)
    
    print("=" * 70)
    print("🏗️  Building Vector Index (LangChain v1.2+)")
    print("=" * 70)
    
    # Process PDF
    print("\n📄 Step 1: Processing PDF...")
    processor = DocumentProcessor()
    chunks = processor.process_pdf(pdf_path)
    
    # Build vector store
    print("\n🔢 Step 2: Creating embeddings...")
    build_vector_store(chunks, "tax_policy_index")
    
    print("\n" + "=" * 70)
    print("✅ SUCCESS!")
    print("=" * 70)
    print(f"\n📊 Statistics:")
    print(f"   - Chunks: {len(chunks)}")
    print(f"   - Source: {Path(pdf_path).name}")
    print(f"\n🚀 Next steps:")
    print(f"   - Streamlit: streamlit run streamlit_app/app.py")
    print(f"   - API: uvicorn api.main:app --reload")
    print(f"   - Test: python scripts/test_query.py")


if __name__ == "__main__":
    main()