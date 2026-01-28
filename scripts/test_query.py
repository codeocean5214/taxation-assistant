"""
Test Query Script - LangChain v1.2+
Interactive CLI for testing RAG system

Usage:
    python scripts/test_query.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from core.rag_engine import initialize_rag_engine


def main():
    print("=" * 70)
    print("🧪 Tax Policy RAG - CLI Tester (LangChain v1.2+)")
    print("=" * 70)
    
    # Initialize
    try:
        print("\n🔧 Initializing RAG engine...")
        engine = initialize_rag_engine()
        print("✅ Ready!\n")
    except FileNotFoundError:
        print("❌ Vector store not found.")
        print("   Run: python scripts/build_index.py <pdf_path>")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        print("   Make sure Ollama is running: ollama serve")
        sys.exit(1)
    
    # Interactive loop
    print("Ask questions (type 'quit' to exit)")
    print("-" * 70)
    
    while True:
        question = input("\n❓ Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not question:
            continue
        
        print("\n🤔 Processing...")
        try:
            result = engine.query(question, return_sources=True)
            
            print("\n💡 Answer:")
            print("-" * 70)
            print(result["answer"])
            
            if result.get("sources"):
                print("\n📚 Sources:")
                print("-" * 70)
                for i, src in enumerate(result["sources"], 1):
                    page = src["metadata"].get("page", "N/A")
                    preview = src["content"][:150].replace("\n", " ")
                    print(f"{i}. Page {page}: {preview}...")
            
            print("\n" + "=" * 70)
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("   Check that Ollama is running: ollama serve")


if __name__ == "__main__":
    main()