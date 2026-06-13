# ============================================
# Test - PDF Study Assistant
# Author: Prateek Kumar Kuntal
# Day 56 - 66 Days GenAI Challenge
# ============================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

def test_imports():
    print("Testing imports...")
    from config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RETRIEVAL
    print(f"  ✅ config — CHUNK_SIZE={CHUNK_SIZE}, OVERLAP={CHUNK_OVERLAP}, TOP_K={TOP_K_RETRIEVAL}")

    from document_processor import DocumentProcessor
    dp = DocumentProcessor()
    print(f"  ✅ DocumentProcessor created")

    from vector_store import VectorStore
    vs = VectorStore()
    print(f"  ✅ VectorStore created")

    from rag_engine import RAGEngine
    rag = RAGEngine()
    print(f"  ✅ RAGEngine created")

    from chatbot import PDFStudyAssistant
    bot = PDFStudyAssistant()
    print(f"  ✅ PDFStudyAssistant created")

    print("\n🎉 All imports passed! Run the app with:")
    print("   streamlit run app.py\n")

if __name__ == "__main__":
    test_imports()
