# ============================================
# Test Script - PDF Study Assistant
# Author: Prateek Kumar Kuntal
# Date: 16 June 2026
# ============================================

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir     = os.path.join(current_dir, "src")
sys.path.insert(0, current_dir)
sys.path.insert(0, src_dir)

from src.chatbot import PDFStudyAssistant


def create_test_document():
    content = """
INTRODUCTION TO MACHINE LEARNING

Chapter 1: What is Machine Learning
Machine learning is a subset of artificial intelligence that enables
computers to learn from data without being explicitly programmed.
There are three main types of machine learning:
1. Supervised learning uses labeled training data
2. Unsupervised learning finds patterns without labels
3. Reinforcement learning trains through rewards

Chapter 2: Key Algorithms
Common supervised learning algorithms include:
- Linear Regression for predicting continuous values
- Logistic Regression for binary classification
- Decision Trees that split data on feature values
- Random Forest which combines multiple decision trees
- Support Vector Machines for classification and regression

Chapter 3: Model Evaluation
Evaluating model performance is critical for production systems.
Key metrics for classification:
- Accuracy measures overall correct predictions
- Precision measures true positives among all positives
- Recall measures true positives among actual positives
- F1 Score is the harmonic mean of precision and recall

For regression tasks use MSE, RMSE, MAE, and R-squared.
Cross validation helps get reliable performance estimates.

Chapter 4: Overfitting and Regularization
Overfitting happens when a model memorizes training data.
It shows high training accuracy but low test accuracy.
Solutions include regularization, dropout, and early stopping.
L1 regularization adds absolute weight penalties.
L2 regularization adds squared weight penalties.
Data augmentation artificially increases training data size.
"""
    os.makedirs("data", exist_ok=True)
    with open("data/test_document.txt", "w") as f:
        f.write(content)
    print("Test document created!")
    return "data/test_document.txt"


def run_tests():
    print("=" * 60)
    print("PDF Study Assistant - Test Run")
    print("=" * 60)

    print("\nInitializing assistant...")
    assistant = PDFStudyAssistant()

    print("\nTest 1: Loading document...")
    doc_path = create_test_document()
    result   = assistant.load_file(doc_path)
    print(f"Result: {result}")
    assert result["success"], "FAILED: Document loading!"
    print("PASSED")

    print("\nTest 2: Basic Q&A...")
    questions = [
        "What are the three types of machine learning?",
        "What is overfitting?",
        "What metrics evaluate classification models?",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        result = assistant.chat(q)
        print(f"A: {result['answer'][:150]}...")
        print(f"Sources: {result['sources']}")
        assert result["answer"], "FAILED: Empty answer!"
    print("\nPASSED")

    print("\nTest 3: Conversation memory...")
    assistant.chat("What is linear regression?")
    result = assistant.chat(
        "Can you give me an example of that?")
    print(f"A: {result['answer'][:100]}...")
    print("PASSED")

    print("\nTest 4: Document summary...")
    summary = assistant.get_document_summary()
    print(f"Summary: {summary[:200]}...")
    assert summary, "FAILED: Empty summary!"
    print("PASSED")

    print("\nTest 5: Statistics...")
    stats = assistant.get_stats()
    print(f"Stats: {stats}")
    print("PASSED")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    print("\nNext step: streamlit run app.py")


if __name__ == "__main__":
    run_tests()