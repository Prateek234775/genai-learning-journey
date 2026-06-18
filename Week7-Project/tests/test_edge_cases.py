# ============================================
# Tests - Edge Cases and Error Handling
# Author: Prateek Kumar Kuntal
# Date: 19 June 2026
# ============================================

import sys
import os
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
src_dir     = os.path.join(parent_dir, "src")
sys.path.insert(0, parent_dir)
sys.path.insert(0, src_dir)

from src.chatbot import PDFStudyAssistant


def test_chat_before_loading_document():
    print("\nTest: Chat before any document loaded")
    assistant = PDFStudyAssistant()
    result = assistant.chat("What is this document about?")
    assert "upload" in result["answer"].lower(), \
        "Should ask user to upload a document"
    print(f"Response: {result['answer']}")
    print("PASSED")


def test_empty_question():
    print("\nTest: Empty question string")
    assistant = PDFStudyAssistant()

    with open("data/edge_test.txt", "w") as f:
        f.write("Sample content for edge case testing of empty questions.")

    assistant.load_file("data/edge_test.txt")
    result = assistant.chat("")
    print(f"Empty question response: {result['answer'][:100]}")
    print("PASSED - did not crash")


def test_very_long_question():
    print("\nTest: Very long question")
    assistant = PDFStudyAssistant()

    with open("data/edge_test2.txt", "w") as f:
        f.write("Machine learning is a method of data analysis "
                "that automates analytical model building.")

    assistant.load_file("data/edge_test2.txt")

    long_question = "What is machine learning " * 50
    result = assistant.chat(long_question)
    assert result["answer"], "Should still return an answer"
    print("PASSED - handled long question")


def test_special_characters_in_question():
    print("\nTest: Special characters in question")
    assistant = PDFStudyAssistant()

    with open("data/edge_test3.txt", "w") as f:
        f.write("Python uses special syntax like @decorators and #comments.")

    assistant.load_file("data/edge_test3.txt")

    special_question = "What about @decorators & #comments?? <test> {json}"
    result = assistant.chat(special_question)
    assert result["answer"], "Should handle special characters"
    print(f"Response: {result['answer'][:100]}...")
    print("PASSED")


def test_question_in_different_language():
    print("\nTest: Question in Hindi")
    assistant = PDFStudyAssistant()

    with open("data/edge_test4.txt", "w") as f:
        f.write("Machine learning helps computers learn patterns from data automatically.")

    assistant.load_file("data/edge_test4.txt")

    hindi_question = "Machine learning kya hai?"
    result = assistant.chat(hindi_question)
    assert result["answer"], "Should handle mixed language questions"
    print(f"Response: {result['answer'][:100]}...")
    print("PASSED")


def test_repeated_rapid_questions():
    print("\nTest: Multiple rapid questions (rate limit check)")
    assistant = PDFStudyAssistant()

    with open("data/edge_test5.txt", "w") as f:
        f.write("Testing rapid fire questions to check rate limiting behavior.")

    assistant.load_file("data/edge_test5.txt")

    questions = [
        "What is this about?",
        "Tell me more.",
        "Why is this important?",
    ]

    for q in questions:
        try:
            result = assistant.chat(q)
            print(f"  Q: {q} -> Got response: {bool(result['answer'])}")
            time.sleep(1)  # be polite to API
        except Exception as e:
            print(f"  Q: {q} -> ERROR: {e}")

    print("PASSED - completed without crashing")


def test_pdf_with_no_extractable_text():
    print("\nTest: Document with minimal content")
    assistant = PDFStudyAssistant()

    with open("data/edge_test6.txt", "w") as f:
        f.write("a")  # single character

    result = assistant.load_file("data/edge_test6.txt")
    print(f"Minimal content load result: {result}")
    print("PASSED - did not crash")


def run_all_tests():
    print("=" * 60)
    print("EDGE CASE AND ERROR HANDLING TESTS")
    print("=" * 60)

    os.makedirs("data", exist_ok=True)

    tests = [
        test_chat_before_loading_document,
        test_empty_question,
        test_very_long_question,
        test_special_characters_in_question,
        test_question_in_different_language,
        test_pdf_with_no_extractable_text,
        test_repeated_rapid_questions,
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