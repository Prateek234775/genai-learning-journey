# ============================================
# Master Test Runner - PDF Study Assistant
# Runs all test suites and prints summary
# Author: Prateek Kumar Kuntal
# Date: 19 June 2026
# ============================================

import sys
import os
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print("=" * 70)
print("PDF STUDY ASSISTANT - FULL TEST SUITE")
print("=" * 70)

start_time = time.time()

print("\n\n>>> RUNNING: Document Processor Tests <<<")
from tests.test_document_processor import run_all_tests as run_doc_tests
run_doc_tests()

print("\n\n>>> RUNNING: Vector Store Tests <<<")
from tests.test_vector_store import run_all_tests as run_vs_tests
run_vs_tests()

print("\n\n>>> RUNNING: Edge Case Tests <<<")
from tests.test_edge_cases import run_all_tests as run_edge_tests
run_edge_tests()

elapsed = time.time() - start_time

print("\n\n" + "=" * 70)
print(f"FULL TEST SUITE COMPLETE in {elapsed:.1f} seconds")
print("=" * 70)
print("\nReview any FAILED or ERROR lines above and fix before deploying.")