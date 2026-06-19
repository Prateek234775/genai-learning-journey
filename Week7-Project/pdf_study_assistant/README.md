# 📚 PDF Study Assistant

> An AI-powered study assistant that lets you chat with your documents, generate quizzes, and create flashcards — all from your own uploaded files.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-green.svg)](https://python.langchain.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red.svg)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/LLM-Groq%20LLaMA3-orange.svg)](https://groq.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 What This Does

Upload any PDF or text document and instantly:
- **Chat with it** — ask questions in natural language
- **Get accurate answers** — grounded only in your document
- **Generate quizzes** — auto MCQs with difficulty levels
- **Create flashcards** — for active recall studying
- **Extract key terms** — instantly identify important concepts
- **Export chat history** — save your Q&A session as markdown

---

## 🖥️ Demo

![PDF Study Assistant Demo](assets/demo.gif)

> Upload → Ask → Get grounded answers with source citations

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│           Streamlit Frontend            │
│   Chat │ Summary │ Quiz │ Flashcards    │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│           PDF Study Assistant           │
│              (chatbot.py)               │
└──────┬──────────────┬───────────────────┘
       │              │
       ▼              ▼
┌────────────┐  ┌─────────────────────────┐
│  Document  │  │       RAG Engine        │
│ Processor  │  │  Query Condensation     │
│ (PDF/TXT)  │  │  Context Retrieval      │
│  Chunking  │  │  Answer Generation      │
└─────┬──────┘  └──────────┬──────────────┘
      │                    │
      ▼                    ▼
┌────────────┐  ┌─────────────────────────┐
│   FAISS    │  │       Groq LLaMA3       │
│  Vector    │  │    (LLM Backend)        │
│   Store    │  │  llama3-8b-8192         │
└────────────┘  └─────────────────────────┘
      │
      ▼
┌────────────┐
│  HuggingFace│
│ Embeddings │
│all-MiniLM  │
└────────────┘
```

---

## ⚡ Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/genai-learning-journey.git
cd genai-learning-journey/Week7-Project/pdf_study_assistant
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up API keys
Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_gemini_key_here
```

Get your free Groq API key at [console.groq.com](https://console.groq.com)

### 4. Run tests
```bash
python run_all_tests.py
```

### 5. Launch the app
```bash
python -m streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| LLM | Groq LLaMA3-8B | Answer generation |
| Embeddings | all-MiniLM-L6-v2 | Text to vectors |
| Vector DB | FAISS | Semantic search |
| Framework | LangChain | RAG orchestration |
| UI | Streamlit | Web interface |
| Document Loading | LangChain Loaders | PDF and TXT parsing |

---

## 📁 Project Structure

```
pdf_study_assistant/
├── src/
│   ├── __init__.py
│   ├── config.py              # All configuration and prompts
│   ├── document_processor.py  # Load and chunk documents
│   ├── vector_store.py        # FAISS vector database
│   ├── rag_engine.py          # RAG chain with Groq
│   ├── chatbot.py             # Main assistant orchestrator
│   └── features.py            # Quiz, flashcards, export
├── tests/
│   ├── test_document_processor.py
│   ├── test_vector_store.py
│   └── test_edge_cases.py
├── data/                      # Document storage
├── app.py                     # Streamlit frontend
├── run_all_tests.py           # Master test runner
├── requirements.txt
├── .env                       # API keys (not committed)
└── README.md
```

---

## 🔍 How RAG Works in This Project

```
1. INDEXING (done once when document is uploaded)
   Document → Split into 500 char chunks
           → Convert to 384-dim embeddings
           → Store in FAISS index

2. RETRIEVAL (every query)
   Question → Convert to embedding
            → Find top 4 similar chunks
            → Return relevant context

3. GENERATION (every query)
   Context + Question + History
            → Groq LLaMA3
            → Grounded answer with source citation
```

---

## ✨ Features

### 💬 Chat
- Multi-turn conversation with memory
- Answers grounded in document content only
- Source citations shown for every answer
- Query condensation for follow-up questions

### 📄 Document Summary
- One-click document summarization
- Bullet-point format for easy reading

### 🎯 Quiz Generator
- Auto-generates MCQs from document
- 3 difficulty levels: easy, medium, hard
- 2 to 10 questions per session

### 🗂️ Flashcards
- Auto-generates front/back flashcards
- Perfect for active recall studying

### 📤 Export
- Download full chat history as markdown
- Timestamped exports

---

## 🧪 Testing

```bash
# Run all tests
python run_all_tests.py

# Run specific test suite
python tests/test_document_processor.py
python tests/test_vector_store.py
python tests/test_edge_cases.py
```

Test coverage includes:
- Document loading (PDF, TXT, empty, missing files)
- Vector store operations (build, search, save, load)
- Edge cases (empty questions, special characters, long inputs)

---

## 🚀 Deployment

### Streamlit Cloud (Free)
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set `GROQ_API_KEY` in Streamlit secrets
5. Deploy!

---

## 📈 Performance

| Metric | Value |
|---|---|
| Chunk size | 500 characters |
| Chunk overlap | 80 characters |
| Embedding dimensions | 384 |
| Top K retrieval | 4 chunks |
| Avg response time | 2-4 seconds |
| Max file size | 10 MB |

---

## 🔧 Configuration

All settings in `src/config.py`:

```python
CHUNK_SIZE       = 500    # Characters per chunk
CHUNK_OVERLAP    = 80     # Overlap between chunks
TOP_K_RETRIEVAL  = 4      # Chunks retrieved per query
LLM_MODEL        = "llama3-8b-8192"  # Groq model
LLM_TEMPERATURE  = 0.3    # Response creativity
```

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push and open a Pull Request

---

## 👤 Author

**Prateek Kumar Kuntal**
- B.Tech AIML — VIT Bhopal
- 56 Day GenAI Engineering Journey — Day 47
- [LinkedIn](www.linkedin.com/in/prateek-kuntal-3886a7281)
- [GitHub](https://github.com/Prateek234775)
- [HuggingFace](https://huggingface.co/prateek5470)

---

## 📄 License

MIT License — feel free to use this project for learning and building.

---

> Built as part of a 56-day structured GenAI engineering self-study program
> covering Python → ML → Deep Learning → NLP → HuggingFace → LangChain → RAG → Deployment