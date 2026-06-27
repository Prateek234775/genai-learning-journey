# ============================================
# HuggingFace Spaces App - PDF Study Assistant
# Author: Prateek Kumar Kuntal
# Date: 24 June 2026
# ============================================

import sys
import os

# Fix paths for HuggingFace Spaces
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(
    current_dir, "..", "Week7-Project", "pdf_study_assistant")
src_dir     = os.path.join(project_dir, "src")

sys.path.insert(0, project_dir)
sys.path.insert(0, src_dir)

# Change working directory
os.chdir(project_dir)

# Now import and run the main app
exec(open(os.path.join(project_dir, "app.py")).read())