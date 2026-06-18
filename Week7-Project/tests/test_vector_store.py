# ============================================
# Tests - Vector Store
# Author: Prateek Kumar Kuntal
# Date: 19 June 2026
# ============================================

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
src_dir     = os.path.join(parent_dir, "src")
sys.path.insert(0, parent_dir)
sys.path.insert(0, src_dir)

from langchain_core.documents import Document
from src.vector_store import VectorStore


def create_sample_docs():
    return [
        Document(
            page_content="Machine learning uses algorithms to learn from data.",
            metadata={"file_name": "ml_basics.txt"}
        ),
        Document(
            page_content="Deep learning uses neural networks with many layers.",
            metadata={"file_name": "dl_basics.txt"}
        ),
        Document(
            page_content="Python is a popular programming language for data science.",
            metadata={"file_name": "python_basics.txt"}
        ),
    ]


def test_build_vector_store():
    print("\nTest: Build vector store from documents")
    vs   = VectorStore()
    docs = create_sample_docs()
    success = vs.build_from_documents(docs)
    assert success, "Should successfully build vector store"
    assert vs.is_ready(), "Vector store should be ready"
    print("PASSED")


def test_search_returns_relevant_results():
    print("\nTest: Search returns relevant results")
    vs   = VectorStore()
    docs = create_sample_docs()
    vs.build_from_documents(docs)

    results = vs.search("neural networks and deep learning", top_k=1)
    assert len(results) == 1, "Should return exactly 1 result"
    assert "neural" in results[0].page_content.lower() or \
           "deep" in results[0].page_content.lower()
    print(f"Top result: {results[0].page_content[:50]}...")
    print("PASSED")


def test_search_empty_vectorstore():
    print("\nTest: Search on empty vector store")
    vs = VectorStore()
    results = vs.search("any query")
    assert results == [], "Empty vector store should return empty list"
    print("PASSED")


def test_add_documents_incrementally():
    print("\nTest: Add documents incrementally")
    vs   = VectorStore()
    docs = create_sample_docs()
    vs.build_from_documents(docs[:2])
    assert vs.doc_count == 2

    vs.add_documents(docs[2:])
    assert vs.doc_count == 3
    print(f"Final doc count: {vs.doc_count}")
    print("PASSED")


def test_save_and_load():
    print("\nTest: Save and load vector store")
    vs   = VectorStore()
    docs = create_sample_docs()
    vs.build_from_documents(docs)

    save_success = vs.save("data/test_vectorstore")
    assert save_success, "Save should succeed"

    vs2 = VectorStore()
    load_success = vs2.load("data/test_vectorstore")
    assert load_success, "Load should succeed"
    assert vs2.is_ready(), "Loaded vector store should be ready"
    print("PASSED")


def test_get_all_sources():
    print("\nTest: Get all sources")
    vs   = VectorStore()
    docs = create_sample_docs()
    vs.build_from_documents(docs)

    sources = vs.get_all_sources()
    assert len(sources) == 3, f"Expected 3 sources, got {len(sources)}"
    print(f"Sources: {sources}")
    print("PASSED")


def test_search_with_scores():
    print("\nTest: Search with similarity scores")
    vs   = VectorStore()
    docs = create_sample_docs()
    vs.build_from_documents(docs)

    results = vs.search_with_scores("machine learning algorithms", top_k=2)
    assert len(results) == 2
    for doc, score in results:
        assert isinstance(score, float), "Score should be a float"
    print(f"Scores: {[round(s, 4) for _, s in results]}")
    print("PASSED")


def run_all_tests():
    print("=" * 60)
    print("VECTOR STORE TESTS")
    print("=" * 60)

    tests = [
        test_build_vector_store,
        test_search_returns_relevant_results,
        test_search_empty_vectorstore,
        test_add_documents_incrementally,
        test_save_and_load,
        test_get_all_sources,
        test_search_with_scores,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()