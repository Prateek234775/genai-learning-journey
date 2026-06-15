# ============================================
# Config - PDF Study Assistant
# Author: Prateek Kumar Kuntal
# Date: 16 June 2026
# ============================================

import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Model Configuration
LLM_MODEL        = "gemini-2.0-flash"
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