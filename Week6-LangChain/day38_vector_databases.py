# ============================================
# DAY 38 - Vector Databases
# FAISS and ChromaDB
# Author: Prateek Kumar Kuntal
# Date: 11 June 2026
# ============================================

import os
import json
import numpy as np
from dotenv import load_dotenv

load_dotenv()


# ------------------------------------------
# PART 1 - WHAT ARE VECTOR DATABASES
# ------------------------------------------

print("===== PART 1: What are Vector Databases =====")

print("""
VECTOR DATABASE:
    Stores data as high dimensional vectors (embeddings)
    Enables semantic similarity search
    Finds items that are similar in meaning not just exact match

TRADITIONAL DATABASE vs VECTOR DATABASE:
    Traditional DB  : "Find all rows where name = 'cat'"
                      Exact keyword matching
    Vector DB       : "Find all items similar to 'feline pet'"
                      Semantic meaning matching
                      'cat', 'kitten', 'tabby' all match!

WHY VECTOR DATABASES MATTER FOR GENAI:
    LLMs cannot access your private data directly
    Vector DB stores your data as embeddings
    When user asks question search vector DB first
    Retrieve relevant chunks and pass to LLM
    LLM answers based on retrieved context
    This is RAG - covered tomorrow!

HOW IT WORKS:
    1. Take your documents (PDFs, text, code)
    2. Split into smaller chunks
    3. Convert each chunk to vector using embedding model
    4. Store vectors in vector database
    5. At query time convert query to vector
    6. Find most similar vectors in database
    7. Return the original text chunks

POPULAR VECTOR DATABASES:
    FAISS       - Facebook, local, fastest, no persistence
    ChromaDB    - open source, easy to use, local + cloud
    Pinecone    - cloud only, production grade, paid
    Weaviate    - open source, full featured
    Qdrant      - open source, high performance
    Milvus      - open source, enterprise scale

    For learning  : FAISS and ChromaDB
    For production: Pinecone or Qdrant
""")


# ------------------------------------------
# PART 2 - EMBEDDINGS
# ------------------------------------------

print("===== PART 2: Embeddings =====")

print("""
EMBEDDINGS:
    Dense vector representations of text
    Capture semantic meaning numerically
    Similar texts have similar vectors
    Measured by cosine similarity or dot product

EMBEDDING MODELS:
    sentence-transformers  - popular, free, local
    OpenAI text-embedding  - powerful, paid API
    Google text-embedding  - powerful, generous free tier
    Cohere embeddings      - multilingual, good quality

DIMENSIONS:
    all-MiniLM-L6-v2       - 384 dimensions (fast, small)
    all-mpnet-base-v2      - 768 dimensions (better quality)
    text-embedding-ada-002 - 1536 dimensions (OpenAI)
    text-embedding-004     - 768 dimensions (Google, free)
""")

from sentence_transformers import SentenceTransformer

print("Loading sentence transformer model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
sentences = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks with many layers",
    "I love eating pizza with extra cheese",
    "Natural language processing deals with text data",
    "The cat sat on the mat near the window",
    "Python is the best language for data science",
    "Neural networks are inspired by the human brain",
    "My favorite food is biryani and dal makhani",
]

print(f"\nGenerating embeddings for {len(sentences)} sentences...")
embeddings = embedding_model.encode(sentences)

print(f"Embedding shape      : {embeddings.shape}")
print(f"Dimensions per vector: {embeddings.shape[1]}")
print(f"\nFirst sentence embedding (first 10 dims):")
print(embeddings[0][:10].round(4))

# Compute similarities
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(embeddings)

print(f"\nSemantic Similarity Matrix (selected pairs):")
print(f"{'Sentence 1':<45} {'Sentence 2':<45} {'Similarity'}")
print("-" * 100)

interesting_pairs = [
    (0, 1),  # ML and Deep Learning - should be high
    (0, 6),  # ML and Neural Networks - should be high
    (2, 7),  # Pizza and Biryani - should be medium (both food)
    (0, 2),  # ML and Pizza - should be low
    (3, 6),  # NLP and Neural Networks - should be medium
    (1, 6),  # Deep Learning and Neural Networks - should be high
]

for i, j in interesting_pairs:
    sim = similarity_matrix[i][j]
    print(f"{sentences[i][:43]:<45} "
          f"{sentences[j][:43]:<45} {sim:.4f}")


# ------------------------------------------
# PART 3 - FAISS
# ------------------------------------------

print("\n===== PART 3: FAISS =====")

print("""
FAISS (Facebook AI Similarity Search):
    Developed by Facebook Research in 2017
    Extremely fast similarity search
    Works entirely in memory (no persistence by default)
    Industry standard for large scale search

FAISS INDEX TYPES:
    IndexFlatL2       - exact search, brute force, small datasets
    IndexFlatIP       - inner product (cosine similarity)
    IndexIVFFlat      - approximate, faster for large datasets
    IndexIVFPQ        - compressed, fastest, slight accuracy loss
    IndexHNSW         - graph based, very fast, good accuracy

FOR LEARNING:
    IndexFlatL2 for small datasets (what we use)
    IndexIVFFlat for production with millions of vectors

FAISS DISTANCE METRICS:
    L2 distance       - Euclidean distance (lower = more similar)
    Inner product     - dot product (higher = more similar)
    For normalized vectors both give same ranking
""")

import faiss

# Create knowledge base
knowledge_base = [
    "Python is a high level programming language used for data science and AI",
    "Machine learning algorithms learn patterns from training data",
    "Deep learning uses multiple layers of neural networks",
    "Gradient descent optimizes model weights by following negative gradient",
    "Overfitting occurs when model memorizes training data instead of learning patterns",
    "Transfer learning reuses pretrained model weights for new tasks",
    "BERT is a bidirectional transformer encoder for NLP tasks",
    "GPT is a decoder only transformer used for text generation",
    "Attention mechanism allows models to focus on relevant input parts",
    "Transformer architecture replaced RNNs for most sequence tasks",
    "LangChain is a framework for building LLM powered applications",
    "RAG combines retrieval with generation for better LLM responses",
    "Vector databases store embeddings for semantic similarity search",
    "Fine tuning adapts pretrained models to specific downstream tasks",
    "Prompt engineering improves LLM outputs without changing model weights",
    "LoRA reduces fine tuning cost by training low rank adapter matrices",
    "ChromaDB is an open source vector database for AI applications",
    "FAISS enables fast approximate nearest neighbor search at scale",
    "Embeddings are dense vector representations of text or images",
    "Cosine similarity measures angle between two vectors in high dimensional space",
]

print(f"Knowledge base size  : {len(knowledge_base)} documents")

# Generate embeddings for knowledge base
print("Generating embeddings...")
kb_embeddings = embedding_model.encode(knowledge_base)
kb_embeddings = kb_embeddings.astype("float32")

# Normalize for cosine similarity
faiss.normalize_L2(kb_embeddings)

print(f"Embeddings shape     : {kb_embeddings.shape}")

# Create FAISS index
dimension = kb_embeddings.shape[1]
index     = faiss.IndexFlatIP(dimension)  # Inner product = cosine for normalized

# Add vectors to index
index.add(kb_embeddings)
print(f"FAISS index created  : {index.ntotal} vectors indexed")

# Search function
def faiss_search(query, index, knowledge_base,
                 embedding_model, top_k=3):
    query_embedding = embedding_model.encode([query])
    query_embedding = query_embedding.astype("float32")
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            results.append({
                "text"      : knowledge_base[idx],
                "score"     : float(score),
                "index"     : int(idx),
            })
    return results

# Test FAISS search
queries = [
    "How do neural networks learn?",
    "What is the best way to use pretrained models?",
    "How does attention work in transformers?",
    "What tools are used for building LLM applications?",
]

print(f"\nFAISS Semantic Search Results:")
print("=" * 70)
for query in queries:
    results = faiss_search(
        query, index, knowledge_base, embedding_model, top_k=3)
    print(f"\nQuery: {query}")
    for i, r in enumerate(results):
        print(f"  {i+1}. [{r['score']:.4f}] {r['text'][:70]}...")

# Save and load FAISS index
print(f"\nSaving FAISS index...")
faiss.write_index(index, "knowledge_base.faiss")
print(f"Index saved to knowledge_base.faiss")

loaded_index = faiss.read_index("knowledge_base.faiss")
print(f"Index loaded. Vectors: {loaded_index.ntotal}")


# ------------------------------------------
# PART 4 - CHROMADB
# ------------------------------------------

print("\n===== PART 4: ChromaDB =====")

print("""
CHROMADB:
    Open source vector database
    Easy to use Python API
    Supports persistence (data saved to disk)
    Built in embedding functions
    Metadata filtering
    Multiple collections

CHROMADB vs FAISS:
    FAISS     - faster, no persistence, lower level API
    ChromaDB  - easier, persistent, metadata, higher level

CHROMADB CONCEPTS:
    Client      - connects to ChromaDB
    Collection  - like a table in regular DB
    Document    - the original text
    Embedding   - vector representation
    Metadata    - additional info (source, date etc)
    ID          - unique identifier per document
""")

import chromadb
from chromadb.utils import embedding_functions

# Create ChromaDB client
print("Creating ChromaDB client...")
chroma_client = chromadb.Client()  # in-memory

# For persistent storage use:
# chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Create embedding function
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2")

# Create collection
collection = chroma_client.create_collection(
    name               = "aiml_knowledge_base",
    embedding_function = sentence_transformer_ef,
    metadata           = {"description": "AIML concepts knowledge base"}
)

print(f"Collection created   : {collection.name}")

# Prepare documents with metadata
documents  = knowledge_base
ids        = [f"doc_{i}" for i in range(len(documents))]
metadatas  = []

for i, doc in enumerate(documents):
    if any(word in doc.lower() for word in
           ["python", "langchain", "chromadb", "faiss", "rag"]):
        category = "tools"
    elif any(word in doc.lower() for word in
             ["bert", "gpt", "transformer", "attention"]):
        category = "models"
    elif any(word in doc.lower() for word in
             ["learning", "training", "gradient", "overfitting"]):
        category = "concepts"
    else:
        category = "general"

    metadatas.append({
        "category"  : category,
        "doc_length": len(doc),
        "doc_index" : i,
    })

# Add documents to collection
collection.add(
    documents = documents,
    ids       = ids,
    metadatas = metadatas,
)

print(f"Documents added      : {collection.count()}")

# Query ChromaDB
def chroma_search(query, collection,
                  top_k=3, filter_category=None):
    where = {"category": filter_category} if filter_category else None

    results = collection.query(
        query_texts  = [query],
        n_results    = top_k,
        where        = where,
    )

    output = []
    for i in range(len(results["documents"][0])):
        output.append({
            "text"    : results["documents"][0][i],
            "distance": results["distances"][0][i],
            "metadata": results["metadatas"][0][i],
            "id"      : results["ids"][0][i],
        })
    return output

# Basic search
print(f"\nChromaDB Semantic Search:")
print("=" * 70)

queries = [
    "How do I fine tune a language model efficiently?",
    "What frameworks help build AI applications?",
    "How does BERT understand bidirectional context?",
]

for query in queries:
    results = chroma_search(query, collection, top_k=2)
    print(f"\nQuery: {query}")
    for r in results:
        print(f"  [{r['distance']:.4f}] [{r['metadata']['category']}] "
              f"{r['text'][:65]}...")

# Filtered search
print(f"\nFiltered Search (tools category only):")
query   = "What tools are used for vector similarity search?"
results = chroma_search(
    query, collection, top_k=3, filter_category="tools")
print(f"Query: {query}")
for r in results:
    print(f"  [{r['distance']:.4f}] {r['text'][:70]}...")

# Update and delete operations
print(f"\nChromaDB CRUD Operations:")

# Update document
collection.update(
    ids       = ["doc_0"],
    documents = ["Python is the most popular programming language for AI, ML, and data science worldwide"],
    metadatas = [{"category": "tools", "doc_length": 80, "doc_index": 0, "updated": True}]
)
print(f"Document updated: doc_0")

# Get specific document
result = collection.get(ids=["doc_0"])
print(f"Retrieved doc_0: {result['documents'][0][:60]}...")

# Collection stats
print(f"\nCollection Statistics:")
print(f"  Name          : {collection.name}")
print(f"  Total docs    : {collection.count()}")
print(f"  Metadata      : {collection.metadata}")


# ------------------------------------------
# PART 5 - LANGCHAIN VECTOR STORES
# ------------------------------------------

print("\n===== PART 5: LangChain Vector Stores =====")

print("""
LANGCHAIN VECTORSTORES:
    LangChain wraps vector databases in unified interface
    Same code works with FAISS, ChromaDB, Pinecone etc
    Easy to switch between databases

LANGCHAIN RETRIEVERS:
    Convert vector store into retriever
    Plugs directly into RAG chains
    Multiple retrieval strategies available

RETRIEVAL STRATEGIES:
    Similarity search     - top K most similar
    MMR (Max Marginal Relevance) - diverse results
    Similarity score threshold   - above threshold only
""")

from langchain_community.vectorstores import FAISS as LangFAISS
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# LangChain embedding model
print("Setting up LangChain embeddings...")
lc_embeddings = HuggingFaceEmbeddings(
    model_name = "all-MiniLM-L6-v2")

# Create LangChain FAISS vector store
print("Creating LangChain FAISS vectorstore...")
lc_faiss = LangFAISS.from_texts(
    texts      = knowledge_base,
    embedding  = lc_embeddings,
)

# Create LangChain Chroma vector store
print("Creating LangChain Chroma vectorstore...")
lc_chroma = Chroma.from_texts(
    texts      = knowledge_base,
    embedding  = lc_embeddings,
    collection_name = "lc_knowledge_base",
)

# Similarity search
query = "How do transformer models process sequences?"

print(f"\nLangChain FAISS similarity search:")
faiss_results = lc_faiss.similarity_search(
    query, k=3)
for doc in faiss_results:
    print(f"  {doc.page_content[:70]}...")

print(f"\nLangChain Chroma similarity search:")
chroma_results = lc_chroma.similarity_search(
    query, k=3)
for doc in chroma_results:
    print(f"  {doc.page_content[:70]}...")

# Similarity search with scores
print(f"\nSimilarity search with scores (FAISS):")
results_with_scores = lc_faiss.similarity_search_with_score(
    query, k=3)
for doc, score in results_with_scores:
    print(f"  [{score:.4f}] {doc.page_content[:65]}...")

# MMR search - diverse results
print(f"\nMMR search (diverse results):")
mmr_results = lc_faiss.max_marginal_relevance_search(
    query, k=3, fetch_k=10)
for doc in mmr_results:
    print(f"  {doc.page_content[:70]}...")

# Convert to retriever
retriever = lc_faiss.as_retriever(
    search_type   = "similarity",
    search_kwargs = {"k": 3}
)

print(f"\nRetriever object: {retriever}")
retrieved_docs = retriever.invoke(query)
print(f"Retrieved {len(retrieved_docs)} documents")
for doc in retrieved_docs:
    print(f"  {doc.page_content[:70]}...")

# Save LangChain FAISS
lc_faiss.save_local("langchain_faiss_index")
print(f"\nLangChain FAISS index saved!")

# Load it back
loaded_faiss = LangFAISS.load_local(
    "langchain_faiss_index",
    lc_embeddings,
    allow_dangerous_deserialization=True
)
print(f"LangChain FAISS index loaded!")


# ------------------------------------------
# PART 6 - DOCUMENT CHUNKING
# ------------------------------------------

print("\n===== PART 6: Document Chunking =====")

print("""
DOCUMENT CHUNKING:
    Split large documents into smaller chunks
    Each chunk becomes one vector in database
    Chunk size affects retrieval quality

CHUNKING STRATEGIES:
    Fixed size chunking  - split every N characters
    Sentence chunking    - split at sentence boundaries
    Recursive chunking   - try different separators in order
    Semantic chunking    - split at semantic boundaries

KEY PARAMETERS:
    chunk_size    - target size of each chunk in characters
    chunk_overlap - overlap between adjacent chunks
                    prevents losing context at boundaries

    chunk_size=500, chunk_overlap=50 is a good default

WHY OVERLAP:
    "...end of chunk 1. Start of chunk 2..."
    Without overlap the boundary context is lost
    With overlap both chunks contain the boundary text
""")

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

# Sample long document
long_document = """
Machine learning is a branch of artificial intelligence and computer science
which focuses on the use of data and algorithms to imitate the way that humans
learn, gradually improving its accuracy. Machine learning is an important component
of the growing field of data science.

Through the use of statistical methods, algorithms are trained to make
classifications or predictions, and to uncover key insights in data mining projects.
These insights subsequently drive decision making within applications and businesses,
ideally impacting key growth metrics.

Deep learning is part of a broader family of machine learning methods based on
artificial neural networks with representation learning. Learning can be supervised,
semi-supervised or unsupervised. Deep learning architectures such as deep neural
networks, recurrent neural networks, convolutional neural networks and transformers
have been applied to fields including computer vision, speech recognition, natural
language processing, machine translation, bioinformatics, drug design, and medical
image analysis where they have produced results comparable to and in some cases
surpassing human expert performance.

The transformer architecture was introduced in 2017 and has since revolutionized
natural language processing. It uses attention mechanisms to process all tokens in
parallel rather than sequentially like RNNs. This parallel processing allows for
much more efficient training on modern GPU hardware. BERT and GPT are both based
on the transformer architecture but serve different purposes.

BERT uses an encoder only architecture and is pretrained using masked language
modeling. This means it reads the entire sequence bidirectionally which makes it
excellent for understanding tasks. GPT uses a decoder only architecture and is
pretrained to predict the next token. This autoregressive approach makes it
excellent for text generation tasks.
"""

# Recursive text splitter (recommended)
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size    = 300,
    chunk_overlap = 50,
    separators    = ["\n\n", "\n", ".", " ", ""],
)

recursive_chunks = recursive_splitter.split_text(long_document)

print(f"Original document    : {len(long_document)} characters")
print(f"Number of chunks     : {len(recursive_chunks)}")
print(f"Avg chunk size       : "
      f"{np.mean([len(c) for c in recursive_chunks]):.0f} chars")

print(f"\nChunks:")
for i, chunk in enumerate(recursive_chunks):
    print(f"\nChunk {i+1} ({len(chunk)} chars):")
    print(f"  {chunk[:100]}...")

# Character splitter
char_splitter = CharacterTextSplitter(
    separator     = "\n\n",
    chunk_size    = 400,
    chunk_overlap = 30,
)

char_chunks = char_splitter.split_text(long_document)
print(f"\nCharacter splitter chunks: {len(char_chunks)}")

# Effect of different chunk sizes
print(f"\nEffect of chunk size on number of chunks:")
print(f"{'Chunk Size':<15} {'Overlap':<12} {'Num Chunks'}")
print("-" * 40)
for size, overlap in [(100, 10), (200, 30), (300, 50), (500, 100)]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap)
    chunks = splitter.split_text(long_document)
    print(f"{size:<15} {overlap:<12} {len(chunks)}")


# ------------------------------------------
# MINI PROJECT - Personal Knowledge Base
# ------------------------------------------

print("\n===== MINI PROJECT: Personal Knowledge Base =====")

class PersonalKnowledgeBase:
    def __init__(self, name="My Knowledge Base"):
        self.name         = name
        self.lc_embeddings= HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2")
        self.text_splitter= RecursiveCharacterTextSplitter(
            chunk_size=200, chunk_overlap=30)
        self.vectorstore  = None
        self.documents    = []
        self.doc_count    = 0

    def add_document(self, text, source="manual"):
        chunks = self.text_splitter.split_text(text)
        metadatas = [
            {"source": source, "chunk_index": i,
             "doc_id": self.doc_count}
            for i in range(len(chunks))
        ]

        if self.vectorstore is None:
            self.vectorstore = Chroma.from_texts(
                texts          = chunks,
                embedding      = self.lc_embeddings,
                metadatas      = metadatas,
                collection_name= "personal_kb",
            )
        else:
            self.vectorstore.add_texts(
                texts     = chunks,
                metadatas = metadatas,
            )

        self.documents.append({
            "source": source,
            "chunks": len(chunks),
            "length": len(text),
        })
        self.doc_count += 1
        print(f"Added document from '{source}' "
              f"({len(chunks)} chunks)")

    def search(self, query, top_k=3):
        if self.vectorstore is None:
            return []

        results = self.vectorstore.similarity_search_with_score(
            query, k=top_k)
        return [
            {
                "text"    : doc.page_content,
                "score"   : score,
                "source"  : doc.metadata.get("source"),
                "chunk"   : doc.metadata.get("chunk_index"),
            }
            for doc, score in results
        ]

    def get_retriever(self, k=3):
        if self.vectorstore is None:
            return None
        return self.vectorstore.as_retriever(
            search_kwargs={"k": k})

    def get_stats(self):
        total_chunks = sum(d["chunks"] for d in self.documents)
        return {
            "total_documents" : len(self.documents),
            "total_chunks"    : total_chunks,
            "documents"       : self.documents,
        }

# Build personal knowledge base
kb = PersonalKnowledgeBase("Prateek AIML Knowledge Base")

# Add multiple documents
docs_to_add = [
    ("LangChain is a framework for developing applications powered by language models. It enables applications that are context-aware and can reason. Key components include models, prompts, chains, memory, agents, and retrievers.", "langchain_notes"),
    ("RAG stands for Retrieval Augmented Generation. It combines the power of retrieval systems with large language models. The process involves indexing documents, retrieving relevant chunks, and generating answers based on context.", "rag_notes"),
    ("Vector databases store embeddings as high dimensional vectors. They enable semantic similarity search. Popular options include FAISS for local use, ChromaDB for open source projects, and Pinecone for production cloud deployments.", "vector_db_notes"),
    ("Prompt engineering is the practice of crafting effective prompts for LLMs. Techniques include zero shot, few shot, chain of thought, and role prompting. Good prompts significantly improve model output quality.", "prompt_engineering_notes"),
    ("Fine tuning adapts pretrained models to specific tasks using labeled data. LoRA and QLoRA are parameter efficient fine tuning methods that reduce compute requirements significantly. They work by adding small trainable matrices to frozen model layers.", "fine_tuning_notes"),
]

print("Building Personal Knowledge Base:")
print("=" * 60)
for text, source in docs_to_add:
    kb.add_document(text, source)

# Search the knowledge base
print(f"\nSearching Knowledge Base:")
test_queries = [
    "How do I build an application with LLMs?",
    "What is the best way to make LLMs use my own data?",
    "How can I reduce the cost of fine tuning large models?",
]

for query in test_queries:
    print(f"\nQuery: {query}")
    results = kb.search(query, top_k=2)
    for r in results:
        print(f"  [{r['score']:.4f}] [{r['source']}] "
              f"{r['text'][:70]}...")

# Stats
stats = kb.get_stats()
print(f"\nKnowledge Base Statistics:")
print(f"  Total documents  : {stats['total_documents']}")
print(f"  Total chunks     : {stats['total_chunks']}")
print(f"  Documents:")
for doc in stats["documents"]:
    print(f"    {doc['source']:<35} "
          f"{doc['chunks']} chunks, {doc['length']} chars")


print("\n===== WHAT I LEARNED TODAY =====")
print("What vector databases are and why they matter")
print("How embeddings represent text as vectors")
print("FAISS - fast local vector search")
print("ChromaDB - easy to use vector database")
print("LangChain vectorstore wrappers")
print("Document chunking strategies")
print("Chunk size and overlap tradeoffs")
print("Retriever interface for RAG chains")
print("MMR search for diverse results")
print("Mini Project - Personal Knowledge Base")
print("\nDay 38 Done! Tomorrow - RAG from Scratch!")