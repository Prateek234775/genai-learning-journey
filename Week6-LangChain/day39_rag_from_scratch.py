# ============================================
# DAY 39 - RAG from Scratch
# Retrieval Augmented Generation
# Author: Prateek Kumar Kuntal
# Date: 12 June 2026
# ============================================

import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model          = "gemini-1.5-flash",
    temperature    = 0.3,
    google_api_key = GOOGLE_API_KEY,
)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name = "all-MiniLM-L6-v2")

print("Gemini and embeddings initialized!")


# ------------------------------------------
# PART 1 - WHAT IS RAG
# ------------------------------------------

print("\n===== PART 1: What is RAG =====")

print("""
RAG (Retrieval Augmented Generation):
    Combines information retrieval with LLM generation
    Allows LLMs to answer questions from your own data
    No fine tuning required

THE PROBLEM RAG SOLVES:
    LLMs have a knowledge cutoff date
    LLMs do not know your private documents
    LLMs sometimes hallucinate (make things up)
    Fine tuning is expensive and slow

HOW RAG WORKS:
    INDEXING PHASE (do once):
    1. Load documents (PDF, text, web pages)
    2. Split into smaller chunks
    3. Convert chunks to embeddings
    4. Store in vector database

    RETRIEVAL AND GENERATION PHASE (every query):
    5. User asks a question
    6. Convert question to embedding
    7. Find most similar chunks in vector database
    8. Include retrieved chunks in LLM prompt as context
    9. LLM generates answer based on context

WHY RAG IS BETTER THAN FINE TUNING FOR MOST CASES:
    No training cost
    Knowledge can be updated instantly
    Sources can be cited
    Works with private data
    Answers are grounded in real documents

REAL WORLD APPLICATIONS:
    Customer support bot on company docs
    Legal document Q and A
    Medical research assistant
    Code documentation search
    Personal knowledge management
""")


# ------------------------------------------
# PART 2 - BUILD RAG STEP BY STEP
# ------------------------------------------

print("===== PART 2: Build RAG Step by Step =====")

# Knowledge base - AIML concepts
documents = [
    """Machine Learning Fundamentals:
Machine learning is a subset of artificial intelligence that enables computers
to learn from data without being explicitly programmed. There are three main
types: supervised learning uses labeled data, unsupervised learning finds
hidden patterns, and reinforcement learning trains agents through rewards.
Key algorithms include linear regression, decision trees, random forests,
and gradient boosting. Model evaluation uses metrics like accuracy, precision,
recall, F1 score, and AUC-ROC.""",

    """Deep Learning and Neural Networks:
Deep learning uses neural networks with multiple layers to learn hierarchical
representations. Each layer learns increasingly abstract features. Activation
functions like ReLU introduce non-linearity. Backpropagation computes gradients
using the chain rule. Batch normalization stabilizes training. Dropout prevents
overfitting. Popular architectures include CNNs for images, RNNs and LSTMs
for sequences, and Transformers for NLP tasks.""",

    """Transformer Architecture:
The transformer architecture introduced in Attention is All You Need 2017
revolutionized NLP. It uses multi-head self-attention to process all tokens
in parallel. Positional encoding adds position information. Each encoder layer
has self-attention and feed-forward sublayers with residual connections and
layer normalization. The decoder additionally has cross-attention over encoder
output. This enables learning long-range dependencies efficiently.""",

    """BERT and GPT Models:
BERT is a bidirectional encoder transformer pretrained with masked language
modeling and next sentence prediction. It excels at understanding tasks like
classification, NER, and question answering. GPT is a decoder-only transformer
pretrained with causal language modeling to predict next tokens. It excels at
text generation, summarization, and conversation. Modern LLMs like ChatGPT
and Claude are based on the GPT architecture with instruction fine tuning.""",

    """LangChain Framework:
LangChain is the most popular framework for building LLM applications. Key
components include Models for LLM wrappers, Prompts for template management,
Chains for composing multiple steps, Memory for conversation history, Tools
for external capabilities, and Agents for autonomous decision making.
LCEL (LangChain Expression Language) uses the pipe operator to compose
components. LangChain supports all major LLMs including OpenAI, Anthropic,
and Google Gemini.""",

    """RAG Systems:
RAG combines retrieval with generation for grounded LLM responses. The
indexing pipeline processes documents through loading, splitting, embedding,
and storing in a vector database. The retrieval pipeline converts queries to
embeddings, searches the vector database, retrieves top-k similar chunks,
and passes them as context to the LLM. Advanced RAG techniques include
query rewriting, hybrid search, reranking, and multi-hop retrieval.""",

    """Vector Databases:
Vector databases store high dimensional embeddings for semantic similarity
search. FAISS by Facebook is fast and runs locally. ChromaDB is open source
and easy to use with persistence. Pinecone is a managed cloud vector database.
Weaviate and Qdrant are enterprise grade options. Key operations include
indexing documents, querying by similarity, and filtering by metadata.
Embedding models like sentence-transformers convert text to vectors.""",

    """Fine Tuning and PEFT:
Fine tuning adapts pretrained models to specific tasks. Full fine tuning
updates all parameters but requires massive compute. PEFT methods reduce
this cost significantly. LoRA adds small trainable rank decomposition matrices
to frozen attention layers. QLoRA extends LoRA with 4-bit quantization to
fine tune billion parameter models on consumer GPUs. The HuggingFace PEFT
library provides easy implementation of these methods.""",

    """Prompt Engineering:
Prompt engineering designs effective inputs to get desired LLM outputs. Zero
shot prompting gives instructions without examples. Few shot prompting provides
input-output examples. Chain of thought asks the model to reason step by step.
System prompts define model persona and rules. Structured output prompts request
JSON or specific formats. Good prompt engineering can dramatically improve
model performance without any fine tuning or additional training cost.""",

    """MLOps and Deployment:
MLOps applies DevOps practices to machine learning. Key components include
data versioning with DVC, experiment tracking with MLflow or Weights and Biases,
model serving with FastAPI or TorchServe, and CI/CD pipelines with GitHub
Actions. Docker containerizes models for consistent deployment. Kubernetes
orchestrates containers at scale. Monitoring tracks model performance and
data drift in production. HuggingFace Spaces provides free model hosting.""",
]

print(f"Knowledge base       : {len(documents)} documents")

# Step 1 - Text Splitting
print("\nStep 1 - Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size    = 300,
    chunk_overlap = 50,
    separators    = ["\n\n", "\n", ".", " ", ""],
)

all_chunks = []
for doc in documents:
    chunks = text_splitter.split_text(doc)
    all_chunks.extend(chunks)

print(f"Total chunks created : {len(all_chunks)}")
print(f"Avg chunk size       : "
      f"{sum(len(c) for c in all_chunks)//len(all_chunks)} chars")

# Step 2 - Create Vector Store
print("\nStep 2 - Creating vector store with embeddings...")
vectorstore = FAISS.from_texts(
    texts     = all_chunks,
    embedding = embeddings,
)
print(f"Vector store created : {len(all_chunks)} vectors indexed")

# Step 3 - Create Retriever
print("\nStep 3 - Creating retriever...")
retriever = vectorstore.as_retriever(
    search_type   = "similarity",
    search_kwargs = {"k": 4},
)
print(f"Retriever ready      : top 4 results per query")

# Step 4 - Test Retrieval
print("\nStep 4 - Testing retrieval...")
test_query = "How does the attention mechanism work?"
retrieved  = retriever.invoke(test_query)

print(f"Query: {test_query}")
print(f"Retrieved {len(retrieved)} chunks:")
for i, doc in enumerate(retrieved):
    print(f"  Chunk {i+1}: {doc.page_content[:80]}...")


# ------------------------------------------
# PART 3 - RAG PROMPT TEMPLATE
# ------------------------------------------

print("\n===== PART 3: RAG Prompt Template =====")

print("""
RAG PROMPT DESIGN:
    Good RAG prompt has 3 parts:
    1. Instructions - how to answer using context
    2. Context      - retrieved documents
    3. Question     - user query

    Key instructions to include:
    - Use only the provided context
    - Say "I don't know" if context does not contain answer
    - Do not make up information
    - Cite sources if possible
""")

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful AI assistant for answering questions about
AI and Machine Learning concepts.

Use ONLY the following context to answer the question.
If the context does not contain enough information to answer
the question, say "I don't have enough information in my
knowledge base to answer this question."

Do not make up information that is not in the context.

Context:
{context}

Question: {question}

Answer:""")

# Format documents helper
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

print("RAG prompt template created!")
print(f"\nTemplate variables: context, question")


# ------------------------------------------
# PART 4 - BASIC RAG CHAIN
# ------------------------------------------

print("\n===== PART 4: Basic RAG Chain =====")

print("""
BASIC RAG CHAIN (LCEL):
    chain = (
        {"context": retriever | format_docs,
         "question": RunnablePassthrough()}
        | prompt
        | llm
        | parser
    )

    RunnablePassthrough passes query through unchanged
    retriever fetches relevant documents
    format_docs combines them into string
    prompt fills template with context and question
    llm generates answer
    parser extracts text from response
""")

# Build the RAG chain
rag_chain = (
    {
        "context" : retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | RAG_PROMPT
    | llm
    | StrOutputParser()
)

print("RAG chain built!")

# Test RAG chain
test_questions = [
    "What is the difference between BERT and GPT?",
    "How does RAG help with LLM hallucinations?",
    "What are the main types of machine learning?",
    "How does LoRA reduce fine tuning costs?",
    "What is LangChain and what are its key components?",
]

print(f"\nRAG Chain Responses:")
print("=" * 70)
for question in test_questions:
    print(f"\nQuestion: {question}")
    answer = rag_chain.invoke(question)
    print(f"Answer  : {answer}")
    print("-" * 70)


# ------------------------------------------
# PART 5 - ADVANCED RAG TECHNIQUES
# ------------------------------------------

print("\n===== PART 5: Advanced RAG Techniques =====")

print("""
ADVANCED RAG TECHNIQUES:

1. QUERY REWRITING:
   Rewrite user query to be more specific
   Improves retrieval accuracy
   "Tell me about BERT" -> "What is BERT architecture and how is it pretrained?"

2. HYPOTHETICAL DOCUMENT EMBEDDING (HyDE):
   Generate hypothetical answer to query
   Use that as search query instead
   Often retrieves more relevant documents

3. MULTI-QUERY RETRIEVAL:
   Generate multiple versions of the query
   Search for each version
   Combine and deduplicate results

4. PARENT DOCUMENT RETRIEVAL:
   Index small chunks for precise retrieval
   Return larger parent chunks for context
   Best of both worlds

5. RERANKING:
   Retrieve more candidates (top 20)
   Rerank using more expensive cross-encoder
   Return top 5 reranked results
   Much higher precision

6. HYBRID SEARCH:
   Combine semantic search with keyword search
   BM25 for keyword + FAISS for semantic
   Combine scores for final ranking
""")

# Technique 1 - Query Rewriting
print("--- Technique 1: Query Rewriting ---")

query_rewrite_prompt = ChatPromptTemplate.from_template("""
You are an expert at improving search queries.
Rewrite the following user question to be more specific and
detailed so it can find the most relevant information.

Original question: {question}

Rewritten question (more specific and searchable):""")

rewrite_chain = (
    query_rewrite_prompt
    | llm
    | StrOutputParser()
)

original_queries = [
    "Tell me about transformers",
    "How does RAG work?",
    "What is fine tuning?",
]

print(f"Query Rewriting Examples:")
for query in original_queries:
    rewritten = rewrite_chain.invoke({"question": query})
    print(f"\nOriginal : {query}")
    print(f"Rewritten: {rewritten.strip()}")


# Technique 2 - Multi Query Retrieval
print(f"\n--- Technique 2: Multi Query Retrieval ---")

multi_query_prompt = ChatPromptTemplate.from_template("""
Generate 3 different versions of the following question
to help retrieve more relevant information.
Return each version on a separate line with no numbering.

Original question: {question}

3 alternative versions:""")

def multi_query_retrieve(question, retriever, llm, top_k=2):
    # Generate multiple queries
    multi_query_chain = (
        multi_query_prompt
        | llm
        | StrOutputParser()
    )
    queries_text = multi_query_chain.invoke(
        {"question": question})
    queries = [q.strip() for q in
               queries_text.strip().split("\n")
               if q.strip()]

    print(f"  Original: {question}")
    print(f"  Generated queries:")
    for q in queries:
        print(f"    - {q}")

    # Retrieve for each query
    all_docs = []
    seen_content = set()

    for query in [question] + queries:
        docs = retriever.invoke(query)
        for doc in docs:
            if doc.page_content not in seen_content:
                all_docs.append(doc)
                seen_content.add(doc.page_content)

    return all_docs[:top_k * 2]

print(f"\nMulti Query Retrieval:")
question = "How do large language models learn from data?"
docs = multi_query_retrieve(
    question, retriever, llm, top_k=3)
print(f"\nRetrieved {len(docs)} unique documents")
for i, doc in enumerate(docs):
    print(f"  Doc {i+1}: {doc.page_content[:70]}...")


# Technique 3 - RAG with Source Citation
print(f"\n--- Technique 3: RAG with Source Citation ---")

# Add metadata to chunks
from langchain.schema import Document

chunked_docs = []
for i, doc_text in enumerate(documents):
    chunks = text_splitter.split_text(doc_text)
    topic  = doc_text.split(":")[0].strip()

    for j, chunk in enumerate(chunks):
        chunked_docs.append(Document(
            page_content = chunk,
            metadata     = {
                "source"     : topic,
                "doc_index"  : i,
                "chunk_index": j,
            }
        ))

# Create vector store with metadata
vectorstore_meta = FAISS.from_documents(
    documents = chunked_docs,
    embedding = embeddings,
)
retriever_meta = vectorstore_meta.as_retriever(
    search_kwargs={"k": 3})

# RAG with citations
CITATION_PROMPT = ChatPromptTemplate.from_template("""
Answer the question using ONLY the provided context.
After your answer cite the source topics you used.

Context:
{context}

Question: {question}

Answer with citations:""")

def format_docs_with_sources(docs):
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        formatted.append(
            f"[Source: {source}]\n{doc.page_content}")
    return "\n\n".join(formatted)

citation_chain = (
    {
        "context" : retriever_meta | format_docs_with_sources,
        "question": RunnablePassthrough(),
    }
    | CITATION_PROMPT
    | llm
    | StrOutputParser()
)

question = "How does attention mechanism relate to transformers?"
print(f"\nQuestion: {question}")
answer = citation_chain.invoke(question)
print(f"Answer with citations:\n{answer}")


# ------------------------------------------
# PART 6 - EVALUATING RAG
# ------------------------------------------

print("\n===== PART 6: Evaluating RAG =====")

print("""
RAG EVALUATION METRICS:

RETRIEVAL METRICS:
    Recall@K     - how many relevant docs retrieved in top K
    Precision@K  - how many retrieved docs are relevant
    MRR          - mean reciprocal rank of first relevant doc

GENERATION METRICS:
    Faithfulness  - is answer grounded in retrieved context?
    Answer Relevance - does answer address the question?
    Context Relevance- are retrieved chunks relevant?

RAGAS FRAMEWORK:
    Popular library for RAG evaluation
    pip install ragas
    Evaluates faithfulness, answer relevance,
    context precision and recall automatically

SIMPLE EVALUATION:
    Ask LLM to judge if answer is grounded in context
    Check if key facts from answer appear in context
""")

def evaluate_rag_response(question, context, answer, llm):
    eval_prompt = ChatPromptTemplate.from_template("""
Evaluate this RAG system response.

Question: {question}

Retrieved Context:
{context}

Generated Answer:
{answer}

Evaluate on these criteria (score 1-5 each):
1. Faithfulness: Is the answer grounded in the context?
2. Relevance: Does the answer address the question?
3. Completeness: Is the answer complete based on context?

Return scores in this exact format:
Faithfulness: [score]/5
Relevance: [score]/5
Completeness: [score]/5
Overall: [average]/5
Brief feedback: [one sentence]""")

    eval_chain = eval_prompt | llm | StrOutputParser()
    return eval_chain.invoke({
        "question": question,
        "context" : context,
        "answer"  : answer,
    })

# Evaluate a RAG response
eval_question = "What is the difference between BERT and GPT?"
eval_context  = format_docs(retriever.invoke(eval_question))
eval_answer   = rag_chain.invoke(eval_question)

print(f"Evaluating RAG response:")
print(f"Question : {eval_question}")
print(f"Answer   : {eval_answer[:100]}...")
print(f"\nEvaluation:")
evaluation = evaluate_rag_response(
    eval_question, eval_context, eval_answer, llm)
print(evaluation)


# ------------------------------------------
# PART 7 - RAG WITH CONVERSATION HISTORY
# ------------------------------------------

print("\n===== PART 7: Conversational RAG =====")

print("""
CONVERSATIONAL RAG:
    Standard RAG is stateless - no memory
    Each question treated independently
    Conversational RAG maintains chat history

    Challenge: Follow up questions need context
    "What is BERT?" -> "How is it different from GPT?"
    Second question needs to know first was about BERT

SOLUTION - QUERY CONDENSATION:
    Given chat history and new question
    Rewrite question to be standalone
    Then run normal RAG on standalone question
""")

from langchain.memory import ConversationBufferMemory

CONDENSE_PROMPT = ChatPromptTemplate.from_template("""
Given the conversation history and a follow up question,
rewrite the follow up question to be a standalone question
that can be understood without the conversation history.

Chat History:
{chat_history}

Follow Up Question: {question}

Standalone Question:""")

CONVERSATIONAL_RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful AI assistant for AI and ML topics.
Use ONLY the context below to answer the question.

Context:
{context}

Question: {question}

Answer:""")

class ConversationalRAG:
    def __init__(self, retriever, llm):
        self.retriever  = retriever
        self.llm        = llm
        self.history    = []
        self.qa_pairs   = []

        self.condense_chain = (
            CONDENSE_PROMPT
            | self.llm
            | StrOutputParser()
        )

    def format_history(self):
        if not self.history:
            return "No previous conversation."
        formatted = []
        for human, ai in self.history:
            formatted.append(f"Human: {human}")
            formatted.append(f"AI: {ai}")
        return "\n".join(formatted)

    def ask(self, question):
        # Condense question if there is history
        if self.history:
            standalone_q = self.condense_chain.invoke({
                "chat_history": self.format_history(),
                "question"    : question,
            })
            print(f"  Condensed: {standalone_q.strip()}")
        else:
            standalone_q = question

        # Retrieve and generate
        docs    = self.retriever.invoke(standalone_q)
        context = format_docs(docs)

        rag_chain = (
            CONVERSATIONAL_RAG_PROMPT
            | self.llm
            | StrOutputParser()
        )
        answer = rag_chain.invoke({
            "context" : context,
            "question": standalone_q,
        })

        # Update history
        self.history.append((question, answer))
        self.qa_pairs.append({
            "question"   : question,
            "standalone" : standalone_q,
            "answer"     : answer,
        })

        return answer

# Test conversational RAG
conv_rag = ConversationalRAG(retriever, llm)

conversation = [
    "What is BERT?",
    "How is it different from GPT?",
    "Which one is better for text generation?",
    "What about for classification tasks?",
]

print("Conversational RAG Session:")
print("=" * 70)
for question in conversation:
    print(f"\nHuman: {question}")
    answer = conv_rag.ask(question)
    print(f"AI    : {answer}")
    print("-" * 70)


# ------------------------------------------
# MINI PROJECT - Full RAG System
# ------------------------------------------

print("\n===== MINI PROJECT: Complete RAG System =====")

class CompleteRAGSystem:
    def __init__(self, llm, embeddings):
        self.llm         = llm
        self.embeddings  = embeddings
        self.vectorstore = None
        self.retriever   = None
        self.history     = []
        self.doc_count   = 0
        self.query_count = 0

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=250, chunk_overlap=40)

    def load_documents(self, documents, source="custom"):
        all_docs = []
        for i, text in enumerate(documents):
            chunks = self.text_splitter.split_text(text)
            for j, chunk in enumerate(chunks):
                from langchain.schema import Document
                all_docs.append(Document(
                    page_content = chunk,
                    metadata     = {
                        "source"     : source,
                        "doc_index"  : i,
                        "chunk_index": j,
                    }
                ))

        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(
                all_docs, self.embeddings)
        else:
            self.vectorstore.add_documents(all_docs)

        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 4})
        self.doc_count += len(documents)

        print(f"Loaded {len(documents)} documents "
              f"({len(all_docs)} chunks) from '{source}'")

    def query(self, question, use_history=True):
        if self.retriever is None:
            return "No documents loaded yet."

        self.query_count += 1

        # Build context from history
        history_text = ""
        if use_history and self.history:
            recent = self.history[-3:]  # last 3 turns
            history_text = "\n".join([
                f"Q: {h['question']}\nA: {h['answer']}"
                for h in recent
            ])

        # Standalone question
        if history_text:
            standalone = (
                CONDENSE_PROMPT
                | self.llm
                | StrOutputParser()
            ).invoke({
                "chat_history": history_text,
                "question"    : question,
            })
        else:
            standalone = question

        # Retrieve
        docs    = self.retriever.invoke(standalone)
        context = "\n\n".join([
            f"[{doc.metadata.get('source', 'unknown')}]\n"
            f"{doc.page_content}"
            for doc in docs
        ])

        # Generate
        prompt = ChatPromptTemplate.from_template("""
You are a knowledgeable AI assistant.
Answer the question using ONLY the provided context.
If the answer is not in the context say so clearly.

Context:
{context}

Question: {question}

Provide a clear and accurate answer:""")

        answer = (prompt | self.llm | StrOutputParser()).invoke({
            "context" : context,
            "question": standalone,
        })

        self.history.append({
            "question": question,
            "answer"  : answer,
        })

        return answer

    def get_stats(self):
        return {
            "documents_loaded" : self.doc_count,
            "queries_answered" : self.query_count,
            "conversation_turns": len(self.history),
        }

# Create and test full RAG system
rag_system = CompleteRAGSystem(llm, embeddings)

print("Loading documents into RAG system...")
rag_system.load_documents(documents, source="aiml_knowledge_base")

print(f"\nQuerying RAG system:")
print("=" * 70)

test_qs = [
    "What makes transformers better than RNNs?",
    "How does vector database help RAG systems?",
    "What is the relationship between LoRA and QLoRA?",
    "What does MLOps involve?",
]

for q in test_qs:
    print(f"\nQ: {q}")
    a = rag_system.query(q)
    print(f"A: {a}")
    print("-" * 70)

stats = rag_system.get_stats()
print(f"\nRAG System Statistics:")
for key, value in stats.items():
    print(f"  {key:<25} : {value}")


print("\n===== WHAT I LEARNED TODAY =====")
print("What RAG is and the problem it solves")
print("RAG pipeline - indexing and retrieval phases")
print("Building RAG chain with LCEL")
print("Query rewriting for better retrieval")
print("Multi query retrieval for diverse results")
print("RAG with source citation")
print("Evaluating RAG responses automatically")
print("Conversational RAG with query condensation")
print("Complete RAG system with all features")
print("\nDay 39 Done! Tomorrow - RAG Chatbot on PDF!")