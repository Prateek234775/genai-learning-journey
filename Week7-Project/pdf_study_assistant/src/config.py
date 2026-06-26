# ============================================
# Config - PDF Study Assistant
# Works locally AND on Streamlit Cloud
# Author: Prateek Kumar Kuntal
# Date: 16 June 2026
# ============================================

import os



def get_api_key(key_name: str) -> str:
    # First try Streamlit secrets (cloud deployment)
    try:
        import streamlit as st
        val = st.secrets.get(key_name, "")
        if val:
            return val
    except Exception:
        pass

    # Then try environment variables (local)
    val = os.environ.get(key_name, "")
    if val:
        return val

    # Finally try .env file manually
    try:
        env_path = os.path.join(
            os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), ".env")
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(key_name + "="):
                        return line.split("=", 1)[1].strip()
    except Exception:
        pass

    return ""


# API Keys
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