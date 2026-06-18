# ============================================
# Tests - Document Processor
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

from src.document_processor import DocumentProcessor


def setup_test_files():
    os.makedirs("data", exist_ok=True)

    with open("data/test_small.txt", "w") as f:
        f.write("This is a short test document for testing.")

    with open("data/test_empty.txt", "w") as f:
        f.write("")

    long_content = "This is a sentence. " * 200
    with open("data/test_large.txt", "w") as f:
        f.write(long_content)


def test_load_valid_text_file():
    print("\nTest: Load valid text file")
    processor = DocumentProcessor()
    docs = processor.load_text("data/test_small.txt")
    assert len(docs) > 0, "Should load at least one document"
    assert docs[0].page_content, "Document should have content"
    print("PASSED")


def test_load_empty_file():
    print("\nTest: Load empty file")
    processor = DocumentProcessor()
    docs = processor.load_text("data/test_empty.txt")
    # Empty file should still load but with empty content
    print(f"Empty file result: {len(docs)} documents")
    print("PASSED")


def test_load_nonexistent_file():
    print("\nTest: Load nonexistent file")
    processor = DocumentProcessor()
    docs = processor.load_text("data/does_not_exist.txt")
    assert docs == [], "Nonexistent file should return empty list"
    print("PASSED")


def test_chunking_large_document():
    print("\nTest: Chunk large document")
    processor = DocumentProcessor()
    docs   = processor.load_text("data/test_large.txt")
    chunks = processor.split_documents(docs)
    assert len(chunks) > 1, "Large document should create multiple chunks"
    print(f"Created {len(chunks)} chunks from large document")
    print("PASSED")


def test_chunking_small_document():
    print("\nTest: Chunk small document")
    processor = DocumentProcessor()
    docs   = processor.load_text("data/test_small.txt")
    chunks = processor.split_documents(docs)
    assert len(chunks) >= 1, "Should create at least one chunk"
    print("PASSED")


def test_unsupported_file_type():
    print("\nTest: Unsupported file type")
    processor = DocumentProcessor()
    docs = processor.process_file("data/test.xyz")
    assert docs == [], "Unsupported file type should return empty list"
    print("PASSED")


def test_metadata_preserved():
    print("\nTest: Metadata preserved in chunks")
    processor = DocumentProcessor()
    chunks = processor.process_file("data/test_small.txt")
    assert all(
        "file_name" in c.metadata for c in chunks
    ), "All chunks should have file_name metadata"
    print("PASSED")


def test_stats_tracking():
    print("\nTest: Stats tracking")
    processor = DocumentProcessor()
    processor.process_file("data/test_small.txt")
    stats = processor.get_stats()
    assert stats["files_loaded"] == 1
    assert stats["total_chunks"] >= 1
    print(f"Stats: {stats}")
    print("PASSED")


def run_all_tests():
    print("=" * 60)
    print("DOCUMENT PROCESSOR TESTS")
    print("=" * 60)

    setup_test_files()

    tests = [
        test_load_valid_text_file,
        test_load_empty_file,
        test_load_nonexistent_file,
        test_chunking_large_document,
        test_chunking_small_document,
        test_unsupported_file_type,
        test_metadata_preserved,
        test_stats_tracking,
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