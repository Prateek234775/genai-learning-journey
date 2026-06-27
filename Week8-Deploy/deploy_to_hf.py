# ============================================
# Deploy to HuggingFace Spaces
# Author: Prateek Kumar Kuntal
# Date: 24 June 2026
# ============================================

from huggingface_hub import HfApi
import os

# Your details
HF_USERNAME  = "prateek5470"  # change this
SPACE_NAME   = "pdf-study-assistant"
REPO_ID      = f"{HF_USERNAME}/{SPACE_NAME}"

# Your project path
PROJECT_PATH = r"E:\prateek\code with prateek java\genai-learning-journey\Week7-Project\pdf_study_assistant"

api = HfApi()

# Files to upload
files_to_upload = [
    ("app.py",                  "app.py"),
    ("requirements.txt",        "requirements.txt"),
    ("src/__init__.py",         "src/__init__.py"),
    ("src/config.py",           "src/config.py"),
    ("src/chatbot.py",          "src/chatbot.py"),
    ("src/document_processor.py", "src/document_processor.py"),
    ("src/vector_store.py",     "src/vector_store.py"),
    ("src/rag_engine.py",       "src/rag_engine.py"),
    ("src/features.py",         "src/features.py"),
]

print(f"Deploying to: {REPO_ID}")
print("=" * 50)

for local_file, remote_file in files_to_upload:
    local_path = os.path.join(PROJECT_PATH, local_file)

    if os.path.exists(local_path):
        print(f"Uploading: {local_file}...")
        api.upload_file(
            path_or_fileobj = local_path,
            path_in_repo    = remote_file,
            repo_id         = REPO_ID,
            repo_type       = "space",
        )
        print(f"  Done: {remote_file}")
    else:
        print(f"  Skipped (not found): {local_path}")

print("=" * 50)
print("Deployment complete!")
print(f"View your Space at:")
print(f"https://huggingface.co/spaces/{REPO_ID}")