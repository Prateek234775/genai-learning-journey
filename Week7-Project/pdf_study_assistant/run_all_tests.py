# ============================================
# Master Test Runner - PDF Study Assistant
# Author: Prateek Kumar Kuntal
# Date: 19 June 2026
# ============================================

import sys
import os
import time

# Fix all paths
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir     = os.path.join(current_dir, "src")
tests_dir   = os.path.join(current_dir, "tests")

sys.path.insert(0, current_dir)
sys.path.insert(0, src_dir)
sys.path.insert(0, tests_dir)

# Change working directory
os.chdir(current_dir)

print("=" * 70)
print("PDF STUDY ASSISTANT - FULL TEST SUITE")
print("=" * 70)
print(f"Running from  : {current_dir}")
print(f"src dir       : {src_dir}")
print(f"tests dir     : {tests_dir}")
print(f"tests exists  : {os.path.exists(tests_dir)}")

start_time = time.time()

# Import test modules directly without package syntax
print("\n\n>>> RUNNING: Document Processor Tests <<<")
try:
    import importlib.util
    spec   = importlib.util.spec_from_file_location(
        "test_document_processor",
        os.path.join(tests_dir, "test_document_processor.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.run_all_tests()
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n\n>>> RUNNING: Vector Store Tests <<<")
try:
    spec   = importlib.util.spec_from_file_location(
        "test_vector_store",
        os.path.join(tests_dir, "test_vector_store.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.run_all_tests()
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n\n>>> RUNNING: Edge Case Tests <<<")
try:
    spec   = importlib.util.spec_from_file_location(
        "test_edge_cases",
        os.path.join(tests_dir, "test_edge_cases.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.run_all_tests()
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

elapsed = time.time() - start_time

print("\n\n" + "=" * 70)
print(f"FULL TEST SUITE COMPLETE in {elapsed:.1f} seconds")
print("=" * 70)