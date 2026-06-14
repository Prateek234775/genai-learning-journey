<div align="center">

# 📚 PDF Study Assistant
### RAG-powered Document Q&A with Google Gemini

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-green?style=for-the-badge)](https://langchain.com)
[![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-orange?style=for-the-badge&logo=google)](https://ai.google.dev)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-purple?style=for-the-badge)](https://faiss.ai)

**Day 56 of 66 Days GenAI Challenge**
*by Prateek Kumar Kuntal*

[Features](#-features) • [Architecture](#-architecture) • [Setup](#-setup) • [Usage](#-usage) • [Tech Stack](#-tech-stack)

</div>

---

## 🎯 What is This?

PDF Study Assistant is an intelligent document Q&A system built on **Retrieval-Augmented Generation (RAG)**. Upload any PDF or text file and instantly start having a conversation with your document — ask questions, get summaries, and extract key insights using the power of Google Gemini 2.5 Flash.

No more skimming through 100-page textbooks. Just ask.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📄 **Multi-format Support** | Upload PDF and TXT files |
| 📚 **Multi-document Chat** | Load multiple files simultaneously |
| 💬 **Conversational Memory** | Maintains context across follow-up questions |
| 📝 **Auto Summarisation** | One-click 5-point document summary |
| 🔍 **Semantic Search** | FAISS vector search finds relevant chunks |
| ⚡ **Fast Responses** | Gemini 2.5 Flash delivers answers in ~2–4 seconds |
| 🖥️ **Clean UI** | Streamlit web interface with dark mode |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                        │
│                  (Streamlit Web App)                     │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                  PDFStudyAssistant                       │
│                    (chatbot.py)                          │
└──────┬─────────────────┬──────────────────┬─────────────┘
       │                 │                  │
       ▼                 ▼                  ▼
┌──────────────┐  ┌─────────────┐  ┌───────────────┐
│   Document   │  │   Vector    │  │   RAG Engine  │
│  Processor   │  │    Store    │  │  (rag_engine) │
│              │  │             │  │               │
│ PyPDFLoader  │  │    FAISS    │  │ Gemini 2.5    │
│ TextLoader   │  │  HuggingFace│  │    Flash      │
│ Text Splitter│  │ Embeddings  │  │ LangChain     │
└──────────────┘  └─────────────┘  └───────────────┘
```

### How It Works

1. **Ingest** — PDF/TXT files are loaded and split into 500-token chunks with 50-token overlap
2. **Embed** — Each chunk is converted to a 384-dimensional vector using `all-MiniLM-L6-v2`
3. **Store** — Vectors are indexed in FAISS for fast similarity search
4. **Retrieve** — User query is embedded and top-5 most relevant chunks are retrieved
5. **Generate** — Gemini 2.5 Flash generates a grounded answer from retrieved context
6. **Remember** — Last 3 exchanges are kept in memory for follow-up questions

---

## 🗂️ Project Structure

```
pdf_study_assistant/
│
├── app.py                      ← Streamlit UI
├── requirements.txt            ← Dependencies
├── .env                        ← API keys (not committed)
├── .env.example                ← Template
├── test_chatbot.py             ← Import smoke test
├── README.md
│
└── src/
    ├── config.py               ← All settings & constants
    ├── document_processor.py   ← PDF/TXT loading & chunking
    ├── vector_store.py         ← FAISS + HuggingFace embeddings
    ├── rag_engine.py           ← Gemini RAG chain + memory
    └── chatbot.py              ← Main assistant orchestrator
```

---

## ⚙️ Setup

### Prerequisites
- Python 3.10+
- Google AI Studio API key → [Get one free here](https://aistudio.google.com/app/apikey)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/genai-learning-journey.git
cd genai-learning-journey/Week7-Project/pdf_study_assistant

# 2. Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your API key
cp .env.example .env
# Open .env and add your Google API key:
# GOOGLE_API_KEY=AIzaSy...
```

### Run

```bash
# Quick test — verify all imports work
python test_chatbot.py

# Launch the app
python -m streamlit run app.py --server.fileWatcherType none
```

Open your browser at **http://localhost:8501** 🚀

---

## 🖥️ Usage

1. **Upload** a PDF or TXT file using the sidebar
2. **Wait** for processing (chunking + embedding takes ~10–30 seconds)
3. **Chat** — ask anything about your document
4. **Summarise** — click "Summarise Documents" for a quick overview
5. **Multi-doc** — upload multiple files and query across all of them

### Example Questions
```
"What is this document about?"
"Explain the concept of machine learning in simple terms"
"What are the key differences between supervised and unsupervised learning?"
"Summarise chapter 3"
"Give me 5 important points from this document"
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **LLM** | Google Gemini 2.5 Flash |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` |
| **Vector Store** | FAISS (Facebook AI Similarity Search) |
| **Framework** | LangChain |
| **UI** | Streamlit |
| **PDF Parsing** | PyPDF |
| **Language** | Python 3.10+ |

---

## 📊 System Components

- ✅ **RAG Pipeline** — end-to-end retrieval-augmented generation
- ✅ **Vector Database Integration** — FAISS with HuggingFace embeddings
- ✅ **Data Preprocessing** — intelligent chunking with overlap
- ✅ **Prompt Engineering** — separate prompts for answering, condensing, and summarising
- ✅ **Frontend / UI** — responsive Streamlit interface
- ✅ **API / Backend** — Google Gemini API via LangChain

---

## 🔧 Configuration

All settings are in `src/config.py`:

```python
LLM_MODEL       = "gemini-2.5-flash"     # Gemini model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"     # HuggingFace embeddings
CHUNK_SIZE      = 500                     # Tokens per chunk
CHUNK_OVERLAP   = 50                      # Overlap between chunks
TOP_K_RETRIEVAL = 5                       # Chunks retrieved per query
```

---

## 🚀 Part of 66 Days GenAI Challenge

This project is **Day 56** of my 66 Days GenAI Challenge — a personal commitment to build one GenAI project every day for 66 days, covering the full spectrum from Python basics to production RAG systems.

| Week | Focus |
|------|-------|
| Week 1 | Python Fundamentals |
| Week 2 | Machine Learning |
| Week 3 | Deep Learning |
| Week 4 | NLP |
| Week 5 | HuggingFace |
| Week 6 | LangChain |
| **Week 7** | **End-to-End GenAI Projects** ← You are here |

---

## 📄 License

MIT License — feel free to use, modify, and build on this project.

---

<div align="center">
Built with ❤️ by <b>Prateek Kumar Kuntal</b><br>
⭐ Star this repo if you found it helpful!
</div>
