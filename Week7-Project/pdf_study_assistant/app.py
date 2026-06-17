# ============================================
# Streamlit App - PDF Study Assistant
# Author: Prateek Kumar Kuntal
# Date: 17 June 2026
# ============================================

import sys
import os
from datetime import datetime
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir     = os.path.join(current_dir, "src")
sys.path.insert(0, current_dir)
sys.path.insert(0, src_dir)

import streamlit as st
from src.chatbot import PDFStudyAssistant


# ------------------------------------------
# PAGE CONFIG
# ------------------------------------------

st.set_page_config(
    page_title = "PDF Study Assistant",
    page_icon  = "📚",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)


# ------------------------------------------
# CUSTOM CSS
# ------------------------------------------

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-badge {
        background-color: #e8f4fd;
        border: 1px solid #1f77b4;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.8rem;
        color: #1f77b4;
        margin-right: 4px;
    }
    .stats-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .chat-info {
        font-size: 0.8rem;
        color: #888;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------
# SESSION STATE INITIALIZATION
# ------------------------------------------

def initialize_session():
    if "assistant" not in st.session_state:
        st.session_state.assistant      = None
    if "messages" not in st.session_state:
        st.session_state.messages       = []
    if "files_loaded" not in st.session_state:
        st.session_state.files_loaded   = []
    if "total_chunks" not in st.session_state:
        st.session_state.total_chunks   = 0
    if "is_initialized" not in st.session_state:
        st.session_state.is_initialized = False

initialize_session()


# ------------------------------------------
# INITIALIZE ASSISTANT
# ------------------------------------------

@st.cache_resource(show_spinner="Loading AI models...")
def load_assistant():
    return PDFStudyAssistant()


# ------------------------------------------
# SIDEBAR
# ------------------------------------------

with st.sidebar:
    st.image("https://img.icons8.com/color/96/book.png",
             width=80)
    st.title("PDF Study Assistant")
    st.caption("Powered by Gemini + LangChain + FAISS")

    st.divider()

    # Initialize button
    if not st.session_state.is_initialized:
        if st.button("Initialize Assistant",
                     type="primary",
                     use_container_width=True):
            with st.spinner("Loading AI models..."):
                try:
                    st.session_state.assistant    = load_assistant()
                    st.session_state.is_initialized = True
                    st.success("Assistant ready!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    else:
        st.success("Assistant is ready!")

    st.divider()

    # File upload section
    st.subheader("Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type            = ["pdf", "txt"],
        accept_multiple_files = True,
        help            = "Upload study materials to chat with",
    )

    if uploaded_files and st.session_state.is_initialized:
        if st.button("Process Documents",
                     type    = "primary",
                     use_container_width = True):
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state.files_loaded:
                    with st.spinner(
                            f"Processing {uploaded_file.name}..."):
                        try:
                            result = st.session_state.assistant.load_from_bytes(
                                uploaded_file.read(),
                                uploaded_file.name,
                            )
                            if result["success"]:
                                st.session_state.files_loaded.append(
                                    uploaded_file.name)
                                st.session_state.total_chunks += result["chunks"]
                                st.success(
                                    f"Loaded: {uploaded_file.name} "
                                    f"({result['chunks']} chunks)")
                            else:
                                st.error(
                                    f"Failed: {uploaded_file.name}")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")

    st.divider()

    # Loaded files display
    if st.session_state.files_loaded:
        st.subheader("Loaded Documents")
        for fname in st.session_state.files_loaded:
            st.markdown(f"- {fname}")
        st.caption(
            f"Total chunks: {st.session_state.total_chunks}")

    st.divider()
    
    # Key terms
    if st.session_state.files_loaded:
        if st.button("Show Key Terms",
                     use_container_width=True):
            with st.spinner("Extracting key terms..."):
                terms = st.session_state.assistant.get_key_terms()
                if terms:
                    st.subheader("Key Terms")
                    terms_html = " ".join([
                        f'<span class="source-badge">{t}</span>'
                        for t in terms
                    ])
                    st.markdown(terms_html, unsafe_allow_html=True)

    # Settings
    st.subheader("Settings")
    top_k = st.slider(
        "Documents to retrieve",
        min_value = 2,
        max_value = 8,
        value     = 4,
        help      = "Higher = more context but slower",
    )

    # Clear chat button
    if st.button("Clear Chat History",
                 use_container_width=True):
        st.session_state.messages = []
        if st.session_state.assistant:
            st.session_state.assistant.clear_chat()
        st.rerun()

    st.divider()
    # Export chat button
    if st.session_state.messages:
        export_data = st.session_state.assistant.export_history(
            "markdown")
        st.download_button(
            label     = "Export Chat History",
            data      = export_data,
            file_name = f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime      = "text/markdown",
            use_container_width = True,
        )
    # Stats
    if (st.session_state.is_initialized and
            st.session_state.assistant):
        st.subheader("Session Stats")
        stats = st.session_state.assistant.get_stats()
        rag   = stats.get("rag_engine", {})
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Queries",
                      rag.get("total_queries", 0))
        with col2:
            st.metric("Files",
                      len(st.session_state.files_loaded))

    st.divider()
    st.caption("Built by Prateek Kumar Kuntal")
    st.caption("56 Day GenAI Journey - Day 44")


# ------------------------------------------
# MAIN CONTENT
# ------------------------------------------

st.markdown(
    '<div class="main-header">📚 PDF Study Assistant</div>',
    unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Upload any document and chat with it using AI</div>',
    unsafe_allow_html=True)


# ------------------------------------------
# NOT INITIALIZED STATE
# ------------------------------------------

if not st.session_state.is_initialized:
    st.info(
        "Click **Initialize Assistant** in the sidebar "
        "to get started.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        ### How it works
        1. Initialize the assistant
        2. Upload your PDF or text files
        3. Click Process Documents
        4. Start asking questions!
        """)
    with col2:
        st.markdown("""
        ### What you can do
        - Ask questions about uploaded docs
        - Get summaries of content
        - Have multi-turn conversations
        - See which source was used
        """)
    with col3:
        st.markdown("""
        ### Tech stack
        - Gemini 1.5 Flash (LLM)
        - FAISS (Vector Database)
        - LangChain (Framework)
        - Streamlit (UI)
        """)
    st.stop()


# ------------------------------------------
# NO DOCUMENTS STATE
# ------------------------------------------

if not st.session_state.files_loaded:
    st.warning(
        "No documents loaded yet. "
        "Upload files in the sidebar and click "
        "Process Documents.")

    # Demo mode with test document
    if st.button("Load Demo Document",
                 type="secondary"):
        demo_content = """
MACHINE LEARNING FUNDAMENTALS

Chapter 1: Introduction
Machine learning enables computers to learn from data.
Three types: supervised, unsupervised, reinforcement learning.
Supervised learning uses labeled data for training.
Unsupervised learning finds hidden patterns in data.
Reinforcement learning trains agents through rewards.

Chapter 2: Key Algorithms
Linear Regression predicts continuous numerical values.
Logistic Regression classifies binary outcomes.
Decision Trees make predictions using feature splits.
Random Forest combines many decision trees together.
Neural Networks learn complex non-linear patterns.

Chapter 3: Evaluation
Accuracy measures overall correct predictions.
Precision measures quality of positive predictions.
Recall measures completeness of positive predictions.
F1 Score balances precision and recall together.
Cross validation gives reliable performance estimates.

Chapter 4: Best Practices
Always split data into train and test sets.
Normalize features before training most models.
Use cross validation for reliable evaluation.
Prevent overfitting with regularization techniques.
Monitor both training and validation performance.
"""
        os.makedirs("data", exist_ok=True)
        demo_path = "data/demo_ml_guide.txt"
        with open(demo_path, "w") as f:
            f.write(demo_content)

        with st.spinner("Loading demo document..."):
            result = st.session_state.assistant.load_file(
                demo_path)
            if result["success"]:
                st.session_state.files_loaded.append(
                    "demo_ml_guide.txt")
                st.session_state.total_chunks += result["chunks"]
                st.success("Demo document loaded!")
                st.rerun()
    st.stop()


# ------------------------------------------
# DOCUMENT SUMMARY TAB AND CHAT TAB
# ------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Chat", "📄 Summary", "🎯 Quiz", "🗂️ Flashcards"
])

with tab2:
    st.subheader("Document Summary")
    if st.button("Generate Summary",
                 type="primary"):
        with st.spinner("Generating summary..."):
            summary = st.session_state.assistant.get_document_summary()
            st.markdown(summary)

with tab3:
    st.subheader("Auto-Generated Quiz")
    col1, col2 = st.columns(2)
    with col1:
        n_questions = st.slider(
            "Number of questions", 2, 10, 3)
    with col2:
        difficulty = st.selectbox(
            "Difficulty", ["easy", "medium", "hard"])

    if st.button("Generate Quiz", type="primary"):
        with st.spinner("Creating quiz from your documents..."):
            quiz = st.session_state.assistant.create_quiz(
                n_questions, difficulty)
            st.markdown(quiz)

with tab4:
    st.subheader("Study Flashcards")
    n_cards = st.slider("Number of flashcards", 3, 10, 5)

    if st.button("Generate Flashcards", type="primary"):
        with st.spinner("Creating flashcards..."):
            cards = st.session_state.assistant.create_flashcards(
                n_cards)

            if cards:
                for i, card in enumerate(cards):
                    with st.expander(
                            f"Card {i+1}: {card.get('front', '')}"):
                        st.write(card.get("back", ""))
            else:
                st.warning(
                    "Could not generate flashcards. Try again.")
                
with tab1:
    # ------------------------------------------
    # CHAT INTERFACE
    # ------------------------------------------

    # Display chat messages
    chat_container = st.container()
    with chat_container:
        if not st.session_state.messages:
            st.info(
                "Ask me anything about your uploaded documents!")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                # Show sources for assistant messages
                if (message["role"] == "assistant" and
                        message.get("sources")):
                    sources_html = " ".join([
                        f'<span class="source-badge">{s}</span>'
                        for s in message["sources"]
                    ])
                    st.markdown(
                        f'<div class="chat-info">Sources: '
                        f'{sources_html}</div>',
                        unsafe_allow_html=True)

                    if message.get("chunks_used"):
                        st.markdown(
                            f'<div class="chat-info">'
                            f'Chunks retrieved: '
                            f'{message["chunks_used"]}</div>',
                            unsafe_allow_html=True)

    # Suggested questions
    if not st.session_state.messages:
        st.subheader("Suggested Questions")
        suggestions = [
            "What are the main topics covered in this document?",
            "Summarize the key concepts in bullet points.",
            "What are the most important things to remember?",
            "Explain the first chapter in simple terms.",
        ]

        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(suggestion,
                             use_container_width=True,
                             key=f"suggest_{i}"):
                    st.session_state.pending_question = suggestion
                    st.rerun()

    # Handle suggested question
    if hasattr(st.session_state, "pending_question"):
        question = st.session_state.pending_question
        del st.session_state.pending_question

        st.session_state.messages.append({
            "role"   : "user",
            "content": question,
        })

        with st.spinner("Thinking..."):
            result = st.session_state.assistant.chat(
                question, top_k=top_k)

        st.session_state.messages.append({
            "role"       : "assistant",
            "content"    : result["answer"],
            "sources"    : result.get("sources", []),
            "chunks_used": result.get("chunks_used", 0),
        })
        st.rerun()

    # Chat input
    if question := st.chat_input(
            "Ask a question about your documents..."):

        # Add user message
        st.session_state.messages.append({
            "role"   : "user",
            "content": question,
        })

        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(question)

        # Generate and show assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                result = st.session_state.assistant.chat(
                    question, top_k=top_k)

            answer  = result["answer"]
            sources = result.get("sources", [])
            chunks  = result.get("chunks_used", 0)

            st.markdown(answer)

            if sources:
                sources_html = " ".join([
                    f'<span class="source-badge">{s}</span>'
                    for s in sources
                ])
                st.markdown(
                    f'<div class="chat-info">Sources: '
                    f'{sources_html}</div>',
                    unsafe_allow_html=True)

            if chunks:
                st.markdown(
                    f'<div class="chat-info">'
                    f'Chunks retrieved: {chunks}</div>',
                    unsafe_allow_html=True)

        # Save assistant message
        st.session_state.messages.append({
            "role"       : "assistant",
            "content"    : answer,
            "sources"    : sources,
            "chunks_used": chunks,
        })