# ============================================
# Config - PDF Study Assistant
# Works locally AND on Streamlit Cloud
# Author: Prateek Kumar Kuntal
# Date: 16 June 2026
# ============================================

import os
from dotenv import load_dotenv

load_dotenv()

# Read API keys — works both locally and on Streamlit Cloud
def get_api_key(key_name: str) -> str:
    # First try Streamlit secrets (cloud deployment)
    try:
        import streamlit as st
        val = st.secrets.get(key_name, "")
        if val:
            return val
    except Exception:
        pass

    # Fall back to environment variables (local)
    return os.getenv(key_name, "")

GROQ_API_KEY   = get_api_key("GROQ_API_KEY")
GOOGLE_API_KEY = get_api_key("GOOGLE_API_KEY")

# Model Configuration
LLM_MODEL        = "llama3-8b-8192"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
LLM_TEMPERATURE  = 0.3

# RAG Configuration
CHUNK_SIZE        = 500
CHUNK_OVERLAP     = 80
TOP_K_RETRIEVAL   = 4
MAX_HISTORY_TURNS = 6

# Vector Store
VECTORSTORE_PATH  = "data/vectorstore"

# App Configuration
APP_TITLE        = "PDF Study Assistant"
APP_SUBTITLE     = "Upload documents and chat with them using AI"
APP_ICON         = "📚"
MAX_FILE_SIZE_MB = 10
SUPPORTED_TYPES  = ["pdf", "txt"]

# Prompt Templates
SYSTEM_PROMPT = """You are a helpful study assistant that answers
questions based on uploaded documents.

Rules:
- Answer ONLY from the provided document context
- If answer is not in documents say so clearly
- Always be concise and accurate
- Cite which part of the document you used
- Use bullet points for lists
- Use simple language suitable for students"""

RAG_PROMPT_TEMPLATE = """
You are a helpful study assistant.
Use ONLY the following context from the uploaded documents to answer.
If the context does not contain the answer say:
"I could not find this information in the uploaded documents."

Document Context:
{context}

Conversation History:
{history}

Student Question: {question}

Helpful Answer:"""

CONDENSE_PROMPT_TEMPLATE = """
Given the conversation history and a new question,
rewrite the question to be standalone and self-contained.

History:
{chat_history}

New Question: {question}

Standalone Question:"""