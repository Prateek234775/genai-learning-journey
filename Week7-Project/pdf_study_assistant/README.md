# 📚 PDF Study Assistant
**Day 56 — 66 Days GenAI Challenge**
*by Prateek Kumar Kuntal*

A Streamlit-based RAG app to chat with your PDF and text documents using Google Gemini.

---

## 🗂️ Project Structure

```
pdf_study_assistant/
├── app.py                  ← Streamlit UI
├── requirements.txt
├── .env.example            ← copy to .env and add your key
├── test_chatbot.py         ← smoke test
└── src/
    ├── config.py           ← all settings
    ├── document_processor.py
    ├── vector_store.py     ← FAISS + HuggingFace embeddings
    ├── rag_engine.py       ← Gemini RAG chain
    └── chatbot.py          ← main assistant class
```

---

## ⚡ Setup

```powershell
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Google API key
copy .env.example .env
# Edit .env and add your GOOGLE_API_KEY

# 4. Test imports
python test_chatbot.py

# 5. Run the app
streamlit run app.py
```

---

## 🔑 Get a Google API Key
1. Go to https://aistudio.google.com/app/apikey
2. Create a new API key
3. Paste it in your `.env` file

---

## 🛠️ Tech Stack
| Component | Library |
|-----------|---------|
| LLM | Google Gemini 1.5 Flash |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | FAISS |
| Framework | LangChain |
| UI | Streamlit |
