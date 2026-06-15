# ============================================
# DAY 40 - RAG Chatbot on PDF Documents
# Author: Prateek Kumar Kuntal
# Date: 13 June 2026
# ============================================

import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model          = "gemini-2.0-flash",
    temperature    = 0.3,
    google_api_key = GOOGLE_API_KEY,
)

embeddings = HuggingFaceEmbeddings(
    model_name = "all-MiniLM-L6-v2")

print("RAG Chatbot initialized!")


# ------------------------------------------
# PART 1 - DOCUMENT LOADERS
# ------------------------------------------

print("\n===== PART 1: Document Loaders =====")

print("""
DOCUMENT LOADERS:
    LangChain has 100+ document loaders
    Load from PDF, Word, CSV, web, databases

COMMON LOADERS:
    PyPDFLoader          - single PDF file
    PyPDFDirectoryLoader - entire folder of PDFs
    TextLoader           - plain text files
    CSVLoader            - CSV files
    WebBaseLoader        - web pages
    UnstructuredLoader   - handles many formats
    GitLoader            - GitHub repositories
    NotionLoader         - Notion pages
    YoutubeLoader        - YouTube transcripts

PDF LOADING OPTIONS:
    pypdf    - simple, good for most PDFs
    pdfminer - better text extraction
    pymupdf  - fastest, best quality
    unstructured - handles complex layouts

FOR TODAY:
    We create our own PDF-like content
    then use TextLoader since PDF needs
    an actual PDF file to load
""")

# Create sample documents to simulate PDF content
sample_documents = {
    "machine_learning_guide.txt": """
MACHINE LEARNING COMPLETE GUIDE

Chapter 1: Introduction to Machine Learning
Machine learning is a branch of artificial intelligence that enables
systems to learn and improve from experience without being explicitly
programmed. It focuses on developing computer programs that can access
data and use it to learn for themselves.

Types of Machine Learning:
1. Supervised Learning - uses labeled training data
   Examples: Classification, Regression
   Algorithms: Linear Regression, Decision Trees, Random Forest, SVM

2. Unsupervised Learning - finds patterns in unlabeled data
   Examples: Clustering, Dimensionality Reduction
   Algorithms: K-Means, DBSCAN, PCA, Autoencoders

3. Reinforcement Learning - learns through rewards and penalties
   Examples: Game playing, Robot navigation
   Algorithms: Q-Learning, PPO, A3C

Chapter 2: Model Evaluation
Evaluation metrics help assess model performance.

For Classification:
- Accuracy: correct predictions / total predictions
- Precision: true positives / (true positives + false positives)
- Recall: true positives / (true positives + false negatives)
- F1 Score: harmonic mean of precision and recall
- AUC-ROC: area under receiver operating characteristic curve

For Regression:
- MSE: mean squared error
- RMSE: root mean squared error
- MAE: mean absolute error
- R squared: coefficient of determination

Chapter 3: Overfitting and Regularization
Overfitting occurs when model learns training data too well.
Signs of overfitting: high training accuracy, low test accuracy.

Solutions:
- Regularization: L1 (Lasso) and L2 (Ridge) penalties
- Dropout: randomly disable neurons during training
- Early stopping: stop training when validation loss increases
- Data augmentation: artificially increase training data
- Cross validation: evaluate on multiple data splits

Chapter 4: Feature Engineering
Feature engineering transforms raw data into model-ready features.

Techniques:
- Normalization: scale features to 0-1 range
- Standardization: zero mean unit variance
- One hot encoding: convert categories to binary columns
- Feature selection: remove irrelevant features
- PCA: reduce dimensions while preserving variance
""",

    "deep_learning_notes.txt": """
DEEP LEARNING COMPLETE NOTES

Section 1: Neural Networks
A neural network consists of layers of interconnected nodes.
Each connection has a weight that is learned during training.

Components:
- Input layer: receives raw data
- Hidden layers: extract features
- Output layer: produces predictions
- Weights: learned parameters
- Biases: offsets for each neuron
- Activation functions: introduce non-linearity

Common Activation Functions:
- ReLU: max(0, x) - most popular for hidden layers
- Sigmoid: 1/(1+e^-x) - for binary classification output
- Softmax: normalized exponentials - for multi-class output
- Tanh: (e^x - e^-x)/(e^x + e^-x) - centered sigmoid

Section 2: Training Process
Training adjusts weights to minimize loss function.

Steps:
1. Forward pass: compute predictions
2. Compute loss: measure prediction error
3. Backward pass: compute gradients using chain rule
4. Update weights: gradient descent step

Optimizers:
- SGD: simple gradient descent
- Adam: adaptive learning rates with momentum
- RMSprop: adaptive learning rates
- AdaGrad: accumulates gradient history

Section 3: Convolutional Neural Networks
CNNs are specialized for processing grid-like data like images.

Key operations:
- Convolution: apply learnable filters to detect features
- Pooling: reduce spatial dimensions (max or average)
- Flattening: convert feature maps to 1D vector
- Fully connected: traditional neural network layers

Popular architectures:
- VGG: simple deep architecture with 3x3 filters
- ResNet: skip connections to train very deep networks
- EfficientNet: scales width depth and resolution together
- Vision Transformer (ViT): transformer applied to image patches

Section 4: Recurrent Neural Networks
RNNs process sequential data by maintaining hidden state.

Types:
- Vanilla RNN: simple recurrence, vanishing gradient problem
- LSTM: long short term memory with 3 gates
- GRU: gated recurrent unit, simplified LSTM
- Bidirectional: processes sequence in both directions

LSTM Gates:
- Forget gate: decides what to remove from cell state
- Input gate: decides what new information to add
- Output gate: decides what to output as hidden state

Section 5: Transformer Architecture
Transformers replaced RNNs for most sequence tasks.

Key innovations:
- Self-attention: each position attends to all others
- Multi-head attention: multiple attention in parallel
- Positional encoding: inject position information
- Residual connections: prevent vanishing gradients
- Layer normalization: stabilize training

The attention formula:
Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V

where Q=query, K=key, V=value, d_k=key dimension
""",

    "genai_engineering.txt": """
GENAI ENGINEERING PRACTICAL GUIDE

Module 1: Large Language Models
Large language models (LLMs) are neural networks trained on
vast amounts of text data with billions of parameters.

Key LLMs:
- GPT-4: OpenAI, most capable, best reasoning
- Claude: Anthropic, safe and helpful, long context
- Gemini: Google, multimodal, free tier available
- LLaMA: Meta, open source, can run locally
- Mistral: French startup, efficient, open source

How LLMs work:
1. Tokenize input text into subword tokens
2. Convert tokens to embeddings
3. Process through transformer layers
4. Generate next token probabilities
5. Sample from distribution to get next token
6. Repeat until done

Module 2: Prompt Engineering
Effective prompting dramatically improves LLM performance.

Techniques:
- Zero shot: just describe the task
- Few shot: provide input-output examples
- Chain of thought: ask to reason step by step
- Role prompting: assign expert persona
- Structured output: request JSON or specific format
- Delimiters: separate sections clearly

System prompt best practices:
- Define role and expertise clearly
- Specify output format requirements
- List constraints and rules explicitly
- Provide examples of good responses

Module 3: RAG Systems
Retrieval Augmented Generation enhances LLMs with external knowledge.

RAG pipeline:
1. Load documents from various sources
2. Split into overlapping chunks
3. Generate embeddings for each chunk
4. Store in vector database
5. At query time retrieve relevant chunks
6. Include chunks as context in LLM prompt
7. Generate grounded answer

Best practices:
- Chunk size: 300-500 characters for most use cases
- Overlap: 10-15% of chunk size
- Top k: retrieve 3-5 chunks per query
- Embedding model: all-MiniLM for speed, ada-002 for quality
- Reranking: improves precision significantly

Module 4: LangChain Framework
LangChain connects LLMs with tools and data.

Core components:
- Models: wrappers for LLM providers
- Prompts: template management
- Chains: compose multiple steps
- Memory: conversation history
- Agents: autonomous tool use
- Retrievers: document retrieval

LCEL (LangChain Expression Language):
chain = prompt | llm | output_parser
Pipe operator composes components naturally.

Module 5: Deployment
Deploying GenAI applications to production.

Options:
- FastAPI: production REST API framework
- Streamlit: quick web app for demos
- Gradio: ML specific UI components
- Docker: containerize application
- HuggingFace Spaces: free hosting

Key considerations:
- API key security: use environment variables
- Rate limiting: respect API quotas
- Caching: cache repeated queries
- Monitoring: track latency and errors
- Cost management: optimize token usage
"""
}

# Save sample documents
print("Creating sample documents...")
os.makedirs("sample_docs", exist_ok=True)

for filename, content in sample_documents.items():
    filepath = f"sample_docs/{filename}"
    with open(filepath, "w") as f:
        f.write(content)
    print(f"  Created: {filepath}")


# ------------------------------------------
# PART 2 - LOAD AND PROCESS DOCUMENTS
# ------------------------------------------

print("\n===== PART 2: Load and Process Documents =====")

def load_text_documents(directory):
    documents = []
    for filepath in Path(directory).glob("*.txt"):
        loader = TextLoader(str(filepath))
        docs   = loader.load()

        # Add source metadata
        for doc in docs:
            doc.metadata["source"] = filepath.name
            doc.metadata["file"]   = str(filepath)
        documents.extend(docs)
        print(f"  Loaded: {filepath.name} "
              f"({len(docs[0].page_content)} chars)")

    return documents

print("Loading documents...")
raw_documents = load_text_documents("sample_docs")

print(f"\nTotal documents loaded: {len(raw_documents)}")
for doc in raw_documents:
    print(f"  {doc.metadata['source']:<40} "
          f"{len(doc.page_content)} chars")

# Split documents
print("\nSplitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size    = 400,
    chunk_overlap = 60,
    separators    = ["\n\n", "\n", ".", " ", ""],
)

chunks = text_splitter.split_documents(raw_documents)

print(f"Total chunks created : {len(chunks)}")
print(f"Avg chunk size       : "
      f"{sum(len(c.page_content) for c in chunks)//len(chunks)} chars")

# Show sample chunks
print(f"\nSample chunks:")
for i, chunk in enumerate(chunks[:3]):
    print(f"\nChunk {i+1}:")
    print(f"  Source  : {chunk.metadata.get('source')}")
    print(f"  Length  : {len(chunk.page_content)} chars")
    print(f"  Content : {chunk.page_content[:100]}...")


# ------------------------------------------
# PART 3 - CREATE VECTOR STORE
# ------------------------------------------

print("\n===== PART 3: Create Vector Store =====")

print("Creating FAISS vector store from documents...")
vectorstore = FAISS.from_documents(
    documents = chunks,
    embedding = embeddings,
)

# Save vector store
vectorstore.save_local("pdf_vectorstore")
print(f"Vector store saved   : pdf_vectorstore/")
print(f"Total vectors        : {len(chunks)}")

# Test retrieval
retriever = vectorstore.as_retriever(
    search_type   = "similarity",
    search_kwargs = {"k": 4},
)

test_query = "What are the types of machine learning?"
results    = retriever.invoke(test_query)

print(f"\nTest retrieval:")
print(f"Query: {test_query}")
print(f"Retrieved {len(results)} chunks:")
for r in results:
    print(f"  [{r.metadata.get('source')}] "
          f"{r.page_content[:70]}...")


# ------------------------------------------
# PART 4 - BUILD PDF RAG CHATBOT
# ------------------------------------------

print("\n===== PART 4: Build PDF RAG Chatbot =====")

# RAG prompt
PDF_RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful study assistant that answers questions
about machine learning, deep learning, and GenAI engineering.

Use ONLY the provided context to answer questions.
If the answer is not in the context say:
"I could not find information about that in the documents."

Always mention which document the information came from.

Context from documents:
{context}

Conversation history:
{history}

Current question: {question}

Helpful answer:""")

def format_docs_with_source(docs):
    formatted = []
    for doc in docs:
        source  = doc.metadata.get("source", "unknown")
        content = doc.page_content
        formatted.append(f"[From: {source}]\n{content}")
    return "\n\n---\n\n".join(formatted)


class PDFRagChatbot:
    def __init__(self, vectorstore, llm,
                 top_k=4, max_history=6):
        self.vectorstore = vectorstore
        self.llm         = llm
        self.top_k       = top_k
        self.max_history = max_history
        self.history     = []
        self.query_count = 0
        self.sources_used= set()

        self.retriever   = vectorstore.as_retriever(
            search_type   = "similarity",
            search_kwargs = {"k": top_k},
        )

    def format_history(self):
        if not self.history:
            return "No previous messages."
        recent = self.history[-self.max_history:]
        lines  = []
        for turn in recent:
            lines.append(f"User: {turn['question']}")
            lines.append(f"Assistant: {turn['answer'][:100]}...")
        return "\n".join(lines)

    def get_relevant_docs(self, question):
        return self.retriever.invoke(question)

    def answer(self, question):
        self.query_count += 1

        # Retrieve relevant docs
        docs    = self.get_relevant_docs(question)
        context = format_docs_with_source(docs)

        # Track sources
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            self.sources_used.add(source)

        # Generate answer
        chain  = PDF_RAG_PROMPT | self.llm | StrOutputParser()
        answer = chain.invoke({
            "context" : context,
            "history" : self.format_history(),
            "question": question,
        })

        # Store in history
        self.history.append({
            "question": question,
            "answer"  : answer,
            "sources" : [d.metadata.get("source")
                         for d in docs],
        })

        return {
            "answer" : answer,
            "sources": list(set(
                d.metadata.get("source") for d in docs)),
            "chunks_retrieved": len(docs),
        }

    def get_stats(self):
        return {
            "total_queries"    : self.query_count,
            "conversation_turns": len(self.history),
            "sources_consulted": list(self.sources_used),
        }

    def clear_history(self):
        self.history = []
        print("Conversation history cleared!")


# Create chatbot
chatbot = PDFRagChatbot(
    vectorstore = vectorstore,
    llm         = llm,
    top_k       = 4,
    max_history = 6,
)

print("PDF RAG Chatbot created!")

# Test chatbot
test_questions = [
    "What are the three types of machine learning?",
    "How does the LSTM solve the vanishing gradient problem?",
    "What is the difference between precision and recall?",
    "What are the best practices for RAG chunk size?",
    "How does the transformer attention mechanism work?",
    "What tools can I use to deploy a GenAI application?",
]

print(f"\nChatbot Q&A Session:")
print("=" * 70)
for question in test_questions:
    print(f"\nUser      : {question}")
    result = chatbot.answer(question)
    print(f"Assistant : {result['answer']}")
    print(f"Sources   : {result['sources']}")
    print(f"Chunks    : {result['chunks_retrieved']}")
    print("-" * 70)
    time.sleep(1)  # avoid rate limiting


# ------------------------------------------
# PART 5 - ADVANCED CHATBOT FEATURES
# ------------------------------------------

print("\n===== PART 5: Advanced Chatbot Features =====")

# Feature 1 - Document Summary
def summarize_document(doc_name, vectorstore, llm):
    # Get chunks from specific document
    docs = vectorstore.similarity_search(
        "summary overview introduction",
        k=6,
        filter={"source": doc_name}
    )

    if not docs:
        return f"No content found for {doc_name}"

    context = "\n\n".join(d.page_content for d in docs)

    prompt = ChatPromptTemplate.from_template("""
Provide a comprehensive summary of this document.
Include main topics, key concepts, and important points.

Document content:
{context}

Summary:""")

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context})

print("Document Summaries:")
for doc_name in sample_documents.keys():
    print(f"\nSummarizing: {doc_name}")
    summary = summarize_document(
        doc_name, vectorstore, llm)
    print(f"Summary: {summary[:200]}...")
    time.sleep(1)


# Feature 2 - Topic Extraction
def extract_topics(vectorstore, llm, n_topics=5):
    # Sample random chunks
    all_docs = vectorstore.similarity_search(
        "main topics concepts",
        k=10
    )
    context = "\n\n".join(d.page_content for d in all_docs)

    prompt = ChatPromptTemplate.from_template("""
Extract the {n_topics} main topics from this content.
Return as a numbered list with one line description each.

Content:
{context}

Main topics:""")

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "context": context,
        "n_topics": n_topics
    })

print(f"\nMain Topics in Knowledge Base:")
topics = extract_topics(vectorstore, llm)
print(topics)
time.sleep(1)


# Feature 3 - Quiz Generator
def generate_quiz(topic, vectorstore, llm, n_questions=3):
    docs    = vectorstore.similarity_search(topic, k=4)
    context = "\n\n".join(d.page_content for d in docs)

    prompt = ChatPromptTemplate.from_template("""
Generate {n_questions} multiple choice questions about {topic}.
Base questions ONLY on the provided context.

Format each question as:
Q: [question]
A) [option]
B) [option]
C) [option]
D) [option]
Answer: [correct letter]

Context:
{context}

Quiz questions:""")

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "topic"      : topic,
        "n_questions": n_questions,
        "context"    : context,
    })

print(f"\nAuto-generated Quiz:")
quiz = generate_quiz(
    "neural network training",
    vectorstore, llm, n_questions=2)
print(quiz)
time.sleep(1)


# ------------------------------------------
# PART 6 - BUILD STREAMLIT APP
# ------------------------------------------

print("\n===== PART 6: Streamlit App Code =====")

streamlit_code = '''# ============================================
# Streamlit PDF RAG Chatbot
# Save as: streamlit_app.py
# Run with: streamlit run streamlit_app.py
# Author: Prateek Kumar Kuntal
# ============================================

import os
import time
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Page config
st.set_page_config(
    page_title = "PDF RAG Chatbot",
    page_icon  = "📚",
    layout     = "wide"
)

st.title("📚 PDF RAG Chatbot")
st.subtitle("Ask questions about your documents powered by Gemini")

# Sidebar
with st.sidebar:
    st.header("Settings")
    top_k       = st.slider("Documents to retrieve", 2, 8, 4)
    temperature = st.slider("Response creativity", 0.0, 1.0, 0.3)
    st.divider()
    st.header("About")
    st.info(
        "Upload text files and ask questions. "
        "The chatbot retrieves relevant content "
        "and answers using Gemini AI."
    )

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Initialize models
@st.cache_resource
def load_models():
    llm = ChatGoogleGenerativeAI(
        model          = "gemini-1.5-flash",
        temperature    = 0.3,
        google_api_key = os.getenv("GOOGLE_API_KEY"),
    )
    embed = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2")
    return llm, embed

llm, embed = load_models()

# File upload section
st.header("Upload Documents")
uploaded_files = st.file_uploader(
    "Upload text files",
    type=["txt"],
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("Process Documents"):
        with st.spinner("Processing documents..."):
            all_chunks = []
            splitter   = RecursiveCharacterTextSplitter(
                chunk_size=400, chunk_overlap=60)

            for uploaded_file in uploaded_files:
                content = uploaded_file.read().decode("utf-8")
                chunks  = splitter.split_text(content)

                from langchain.schema import Document
                for chunk in chunks:
                    all_chunks.append(Document(
                        page_content = chunk,
                        metadata     = {"source": uploaded_file.name}
                    ))

            st.session_state.vectorstore = (
                FAISS.from_documents(all_chunks, embed))
            st.success(
                f"Processed {len(uploaded_files)} files "
                f"into {len(all_chunks)} chunks!")

# Use existing vectorstore if available
if st.session_state.vectorstore is None:
    try:
        st.session_state.vectorstore = FAISS.load_local(
            "pdf_vectorstore", embed,
            allow_dangerous_deserialization=True)
        st.info("Using existing knowledge base!")
    except Exception:
        st.warning("Please upload documents to get started.")

# Chat interface
st.divider()
st.header("Chat with your Documents")

# Show chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("sources"):
            st.caption(f"Sources: {message['sources']}")

# Chat input
if question := st.chat_input("Ask a question about your documents"):
    if st.session_state.vectorstore is None:
        st.error("Please upload documents first!")
    else:
        # Show user message
        st.chat_message("user").write(question)
        st.session_state.messages.append({
            "role"   : "user",
            "content": question
        })

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                retriever = st.session_state.vectorstore.as_retriever(
                    search_kwargs={"k": top_k})
                docs      = retriever.invoke(question)
                context   = "\\n\\n".join([
                    f"[{d.metadata.get(\'source\')}]\\n{d.page_content}"
                    for d in docs
                ])
                sources = list(set(
                    d.metadata.get("source") for d in docs))

                prompt = ChatPromptTemplate.from_template(
                    "Answer using ONLY this context:\\n{context}"
                    "\\n\\nQuestion: {question}\\n\\nAnswer:")

                chain  = prompt | llm | StrOutputParser()
                answer = chain.invoke({
                    "context" : context,
                    "question": question,
                })

                st.write(answer)
                st.caption(f"Sources: {sources}")

                st.session_state.messages.append({
                    "role"   : "assistant",
                    "content": answer,
                    "sources": str(sources),
                })

# Clear chat button
if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()
'''

# Save Streamlit app
with open("streamlit_app.py", "w") as f:
    f.write(streamlit_code)

print("Streamlit app saved to: streamlit_app.py")
print("\nTo run the Streamlit app:")
print("  streamlit run streamlit_app.py")
print("\nThe app will open in your browser at:")
print("  http://localhost:8501")


# ------------------------------------------
# MINI PROJECT - Complete PDF Assistant
# ------------------------------------------

print("\n===== MINI PROJECT: Complete PDF Assistant =====")

class CompletePDFAssistant:
    def __init__(self, llm, embeddings):
        self.llm          = llm
        self.embeddings   = embeddings
        self.vectorstore  = None
        self.chatbot      = None
        self.loaded_files = []

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400, chunk_overlap=60)

    def load_directory(self, directory):
        print(f"Loading documents from {directory}...")
        all_chunks = []

        for filepath in Path(directory).glob("*.txt"):
            loader  = TextLoader(str(filepath))
            docs    = loader.load()
            chunks  = self.text_splitter.split_documents(docs)

            for chunk in chunks:
                chunk.metadata["source"] = filepath.name

            all_chunks.extend(chunks)
            self.loaded_files.append(filepath.name)
            print(f"  Loaded: {filepath.name} "
                  f"({len(chunks)} chunks)")

        self.vectorstore = FAISS.from_documents(
            all_chunks, self.embeddings)
        self.chatbot     = PDFRagChatbot(
            self.vectorstore, self.llm)

        print(f"Total: {len(all_chunks)} chunks indexed")
        return len(all_chunks)

    def chat(self, question):
        if self.chatbot is None:
            return "Please load documents first."
        return self.chatbot.answer(question)

    def search(self, query, top_k=5):
        if self.vectorstore is None:
            return []
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": top_k})
        return retriever.invoke(query)

    def summarize_all(self):
        if self.vectorstore is None:
            return "No documents loaded."

        docs    = self.vectorstore.similarity_search(
            "main topics overview", k=8)
        context = "\n\n".join(d.page_content for d in docs)

        prompt  = ChatPromptTemplate.from_template("""
Provide a comprehensive overview of all topics
covered in the knowledge base.
List main subjects and key points.

Content sample:
{context}

Overview:""")

        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context})

    def save(self, path="saved_assistant"):
        if self.vectorstore:
            self.vectorstore.save_local(path)
            print(f"Assistant saved to {path}/")

    def load(self, path="saved_assistant"):
        self.vectorstore = FAISS.load_local(
            path, self.embeddings,
            allow_dangerous_deserialization=True)
        self.chatbot     = PDFRagChatbot(
            self.vectorstore, self.llm)
        print(f"Assistant loaded from {path}/")

    def get_stats(self):
        stats = {
            "files_loaded": self.loaded_files,
        }
        if self.chatbot:
            stats.update(self.chatbot.get_stats())
        return stats

# Run complete assistant
assistant = CompletePDFAssistant(llm, embeddings)
assistant.load_directory("sample_docs")

print(f"\nKnowledge Base Overview:")
overview = assistant.summarize_all()
print(overview)
time.sleep(1)

print(f"\nInteractive Q&A:")
print("=" * 70)

final_questions = [
    "What evaluation metrics are used for classification?",
    "Explain the difference between LSTM and GRU.",
    "What are the best practices for prompt engineering?",
    "How do I deploy a GenAI application?",
]

for q in final_questions:
    print(f"\nQ: {q}")
    result = assistant.chat(q)
    print(f"A: {result['answer']}")
    print(f"Source: {result['sources']}")
    print("-" * 70)
    time.sleep(1)

# Save assistant
assistant.save("saved_assistant")

stats = assistant.get_stats()
print(f"\nFinal Statistics:")
print(f"  Files loaded     : {stats['files_loaded']}")
print(f"  Total queries    : {stats.get('total_queries', 0)}")
print(f"  Sources used     : "
      f"{stats.get('sources_consulted', [])}")


print("\n===== WHAT I LEARNED TODAY =====")
print("Document loaders for PDF and text files")
print("Loading and chunking real documents")
print("Building PDF RAG chatbot with conversation history")
print("Document summarization with RAG")
print("Topic extraction from knowledge base")
print("Auto quiz generation from documents")
print("Building Streamlit web app for chatbot")
print("Complete PDF assistant with save and load")
print("\nDay 40 Done! Tomorrow - LangChain Agents!")

print(f"\nFinal Statistixs:")
print(f"  Files loader    : {stats['files_loaded']}")
