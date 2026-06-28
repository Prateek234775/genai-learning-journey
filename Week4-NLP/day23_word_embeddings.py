# ============================================
# DAY 23 - Word Embeddings
# Word2Vec, GloVe, FastText
# Author: Prateek Kumar Kuntal
# Date: 27 May 2026
# ============================================

import numpy as np
from gensim.models import Word2Vec, FastText
from sklearn.metrics.pairwise import cosine_similarity


# ------------------------------------------
# PART 1 - WHAT ARE WORD EMBEDDINGS
# ------------------------------------------

print("===== PART 1: What are Word Embeddings =====")

print("""
WORD EMBEDDINGS:
    Represent words as dense vectors of numbers
    Similar words have similar vectors
    Capture semantic meaning and relationships

PROBLEM WITH BAG OF WORDS AND TF-IDF:
    "king" and "queen" are treated as completely different
    No notion of similarity between words
    Vectors are sparse (mostly zeros)
    Cannot capture context or meaning

WORD EMBEDDINGS SOLVE THIS:
    "king"  -> [0.2, 0.8, -0.3, 0.5, ...]
    "queen" -> [0.2, 0.7, -0.2, 0.6, ...]
    These vectors are close because meaning is similar

FAMOUS EXAMPLE:
    king - man + woman = queen
    paris - france + italy = rome
    Arithmetic works on word meanings!

DIMENSIONS:
    Each word becomes a vector of 50, 100, 300 numbers
    These numbers are learned from text data
    Each dimension captures some aspect of meaning

USED IN:
    Every modern NLP model
    Search engines
    Recommendation systems
    Chatbots and LLMs
""")


# ------------------------------------------
# PART 2 - WORD VECTORS FROM SCRATCH
# ------------------------------------------

print("===== PART 2: Word Vectors from Scratch =====")

print("""
BEFORE Word2Vec - One Hot Encoding:
    Vocabulary of 10000 words
    Each word = vector of 10000 numbers
    Only one position is 1, rest are 0

    "cat" -> [0,0,0,1,0,0,...,0]
    "dog" -> [0,0,0,0,1,0,...,0]

    No similarity information
    Very high dimensional and sparse

AFTER Word2Vec:
    Each word = vector of 300 numbers
    All numbers are meaningful
    Similar words have similar vectors
    Dense and information-rich
""")

# Manual one hot encoding to show the problem
vocabulary = ["king", "queen", "man", "woman",
              "paris", "france", "rome", "italy",
              "cat", "dog"]

def one_hot(word, vocab):
    vector = [0] * len(vocab)
    if word in vocab:
        vector[vocab.index(word)] = 1
    return vector

print("One Hot Encoding:")
print(f"{'Word':<10} {'Vector'}")
print("-" * 50)
for word in vocabulary[:5]:
    vec = one_hot(word, vocabulary)
    print(f"{word:<10} {vec}")

# Cosine similarity between one hot vectors
king_oh  = np.array(one_hot("king",  vocabulary))
queen_oh = np.array(one_hot("queen", vocabulary))
man_oh   = np.array(one_hot("man",   vocabulary))

sim_kq = cosine_similarity([king_oh], [queen_oh])[0][0]
sim_km = cosine_similarity([king_oh], [man_oh])[0][0]

print(f"\nOne Hot Similarity:")
print(f"king vs queen    : {sim_kq:.4f}  (should be high)")
print(f"king vs man      : {sim_km:.4f}  (should be medium)")
print(f"Problem - both are 0! One hot has no similarity info")


# ------------------------------------------
# PART 3 - WORD2VEC
# ------------------------------------------

print("\n===== PART 3: Word2Vec =====")

print("""
WORD2VEC:
    Introduced by Google in 2013
    Learns word vectors from large text corpus
    Uses neural network to predict context

TWO ARCHITECTURES:
    CBOW (Continuous Bag of Words):
        Predict center word from context words
        Context: "The ___ sat on mat"
        Predict: "cat"

    Skip-Gram:
        Predict context words from center word
        Input: "cat"
        Predict: "The", "sat", "on", "mat"
        Better for rare words

KEY PARAMETERS:
    vector_size - dimensions of word vector (100-300)
    window      - context window size (5-10)
    min_count   - ignore words with frequency below this
    sg          - 0 for CBOW, 1 for Skip-Gram
    epochs      - training iterations
""")

# Training corpus
sentences = [
    ["king", "rules", "the", "kingdom", "with", "power"],
    ["queen", "rules", "the", "kingdom", "with", "grace"],
    ["king", "and", "queen", "are", "royal", "family"],
    ["man", "works", "hard", "every", "day"],
    ["woman", "works", "hard", "every", "day"],
    ["man", "and", "woman", "are", "human", "beings"],
    ["paris", "is", "the", "capital", "of", "france"],
    ["rome", "is", "the", "capital", "of", "italy"],
    ["france", "and", "italy", "are", "european", "countries"],
    ["cat", "and", "dog", "are", "common", "pets"],
    ["cat", "meows", "and", "dog", "barks"],
    ["deep", "learning", "is", "subset", "of", "machine", "learning"],
    ["machine", "learning", "uses", "algorithms", "to", "learn"],
    ["neural", "networks", "are", "used", "in", "deep", "learning"],
    ["python", "is", "popular", "language", "for", "machine", "learning"],
    ["data", "science", "uses", "statistics", "and", "machine", "learning"],
    ["artificial", "intelligence", "includes", "machine", "learning"],
    ["natural", "language", "processing", "uses", "deep", "learning"],
    ["word", "embeddings", "represent", "words", "as", "vectors"],
    ["transformers", "use", "attention", "mechanism", "for", "language"],
    ["bert", "and", "gpt", "are", "transformer", "based", "models"],
    ["chatgpt", "uses", "gpt", "architecture", "for", "conversation"],
    ["prateek", "is", "learning", "machine", "learning", "and", "deep", "learning"],
    ["genai", "applications", "use", "large", "language", "models"],
    ["vector", "databases", "store", "word", "embeddings", "for", "search"],
]

# Train Word2Vec
w2v_model = Word2Vec(
    sentences=sentences,
    vector_size=50,
    window=5,
    min_count=1,
    sg=1,           # skip gram
    epochs=200,
    seed=42
)

print(f"Vocabulary size  : {len(w2v_model.wv)}")
print(f"Vector dimensions: {w2v_model.vector_size}")

# Word vectors
print(f"\nWord vector for 'king' (first 10 dims):")
print(w2v_model.wv["king"][:10].round(4))

print(f"\nWord vector for 'queen' (first 10 dims):")
print(w2v_model.wv["queen"][:10].round(4))


# ------------------------------------------
# PART 4 - WORD SIMILARITY
# ------------------------------------------

print("\n===== PART 4: Word Similarity =====")

print("""
COSINE SIMILARITY:
    Measures angle between two vectors
    Range: -1 (opposite) to 1 (identical)
    Similar words have similarity close to 1
""")

def word_similarity(model, word1, word2):
    try:
        sim = model.wv.similarity(word1, word2)
        return sim
    except KeyError as e:
        return f"Word not in vocabulary: {e}"

word_pairs = [
    ("king",    "queen"),
    ("king",    "man"),
    ("man",     "woman"),
    ("paris",   "rome"),
    ("france",  "italy"),
    ("cat",     "dog"),
    ("cat",     "france"),
    ("machine", "learning"),
    ("deep",    "learning"),
    ("bert",    "gpt"),
]

print(f"{'Word 1':<15} {'Word 2':<15} {'Similarity':<12} {'Relationship'}")
print("-" * 60)
for w1, w2 in word_pairs:
    sim = word_similarity(w2v_model, w1, w2)
    if isinstance(sim, float):
        rel = ("Very Similar" if sim > 0.8 else
               "Similar"      if sim > 0.5 else
               "Somewhat"     if sim > 0.2 else
               "Different")
        print(f"{w1:<15} {w2:<15} {sim:<12.4f} {rel}")
    else:
        print(f"{w1:<15} {w2:<15} {sim}")

# Most similar words
print(f"\nMost similar words to 'king':")
similar_to_king = w2v_model.wv.most_similar("king", topn=5)
for word, score in similar_to_king:
    print(f"  {word:<15} : {score:.4f}")

print(f"\nMost similar words to 'learning':")
similar_to_learning = w2v_model.wv.most_similar("learning", topn=5)
for word, score in similar_to_learning:
    print(f"  {word:<15} : {score:.4f}")


# ------------------------------------------
# PART 5 - WORD ARITHMETIC
# ------------------------------------------

print("\n===== PART 5: Word Arithmetic =====")

print("""
WORD ARITHMETIC:
    Famous property of Word2Vec
    Mathematical operations on word vectors
    Capture real world relationships

    king - man + woman = queen
    paris - france + italy = rome

    This shows embeddings capture:
        Gender relationships
        Country capital relationships
        Tense relationships
        And many more
""")

# king - man + woman
try:
    result = w2v_model.wv.most_similar(
        positive=["king", "woman"],
        negative=["man"],
        topn=3
    )
    print("king - man + woman = ?")
    for word, score in result:
        print(f"  {word:<15} : {score:.4f}")
except Exception as e:
    print(f"Error: {e}")

# paris - france + italy
try:
    result = w2v_model.wv.most_similar(
        positive=["paris", "italy"],
        negative=["france"],
        topn=3
    )
    print("\nparis - france + italy = ?")
    for word, score in result:
        print(f"  {word:<15} : {score:.4f}")
except Exception as e:
    print(f"Error: {e}")

# deep - learning + language
try:
    result = w2v_model.wv.most_similar(
        positive=["deep", "language"],
        negative=["learning"],
        topn=3
    )
    print("\ndeep - learning + language = ?")
    for word, score in result:
        print(f"  {word:<15} : {score:.4f}")
except Exception as e:
    print(f"Error: {e}")

print("""
Note: Our corpus is very small (25 sentences)
With millions of sentences results would be perfect
Real Word2Vec trained on Wikipedia or Google News
gives near perfect word arithmetic results
""")


# ------------------------------------------
# PART 6 - GLOVE EMBEDDINGS
# ------------------------------------------

print("===== PART 6: GloVe Embeddings =====")

print("""
GLOVE (Global Vectors for Word Representation):
    Introduced by Stanford in 2014
    Combines global statistics with local context
    Trained on Wikipedia and Common Crawl

HOW IT DIFFERS FROM WORD2VEC:
    Word2Vec  - predicts context (local)
    GloVe     - uses word co-occurrence matrix (global)
    GloVe captures global corpus statistics better

PRETRAINED GLOVE:
    glove.6B.50d.txt   - 50 dimensions
    glove.6B.100d.txt  - 100 dimensions
    glove.6B.300d.txt  - 300 dimensions
    Trained on 6 billion tokens

    Download from: https://nlp.stanford.edu/projects/glove/

SIMULATING GLOVE:
    We will simulate GloVe style embeddings
    since download requires large file
""")

# Simulate GloVe style embeddings
np.random.seed(42)

glove_words = [
    "king", "queen", "man", "woman", "royal",
    "paris", "france", "rome", "italy", "capital",
    "cat", "dog", "pet", "animal",
    "machine", "learning", "deep", "neural", "network",
    "python", "code", "program", "data", "science",
]

# Create embeddings with semantic structure
glove_embeddings = {}
for i, word in enumerate(glove_words):
    np.random.seed(i * 42)
    base_vec = np.random.randn(50)

    # Add semantic grouping
    if word in ["king", "queen", "royal"]:
        base_vec[:5] += [2, 1, 0, 0, 0]
    if word in ["man", "king"]:
        base_vec[5:8] += [1, 0, 0]
    if word in ["woman", "queen"]:
        base_vec[5:8] += [-1, 0, 0]
    if word in ["paris", "rome", "capital"]:
        base_vec[10:15] += [2, 1, 0, 0, 0]
    if word in ["france", "italy"]:
        base_vec[10:15] += [1, 2, 0, 0, 0]
    if word in ["cat", "dog", "pet", "animal"]:
        base_vec[15:20] += [2, 1, 0, 0, 0]
    if word in ["machine", "learning", "deep", "neural", "network"]:
        base_vec[20:25] += [2, 2, 1, 0, 0]

    # Normalize
    base_vec = base_vec / np.linalg.norm(base_vec)
    glove_embeddings[word] = base_vec

print("Simulated GloVe Embeddings:")
print(f"Vocabulary : {len(glove_embeddings)} words")
print(f"Dimensions : 50")

# Test similarity
def glove_similarity(emb, w1, w2):
    v1 = emb[w1].reshape(1, -1)
    v2 = emb[w2].reshape(1, -1)
    return cosine_similarity(v1, v2)[0][0]

print(f"\nGloVe Similarity Scores:")
print(f"{'Pair':<25} {'Similarity'}")
print("-" * 40)
pairs = [
    ("king", "queen"),
    ("king", "man"),
    ("paris", "rome"),
    ("cat", "dog"),
    ("machine", "learning"),
    ("cat", "france"),
]
for w1, w2 in pairs:
    sim = glove_similarity(glove_embeddings, w1, w2)
    print(f"{w1+' vs '+w2:<25} {sim:.4f}")


# ------------------------------------------
# PART 7 - FASTTEXT
# ------------------------------------------

print("\n===== PART 7: FastText =====")

print("""
FASTTEXT:
    Introduced by Facebook in 2016
    Improvement over Word2Vec

KEY DIFFERENCE:
    Word2Vec treats each word as atomic unit
    FastText breaks words into character n-grams

    "playing" -> "pla", "lay", "ayi", "yin", "ing"
    Word vector = average of n-gram vectors

ADVANTAGES:
    Handles out-of-vocabulary words
    Better for rare words
    Works well for morphologically rich languages
    Great for misspelled words

    "playng" (typo) -> similar to "playing"
    Word2Vec would have no vector for this
    FastText still gives a reasonable vector
""")

# Train FastText
ft_model = FastText(
    sentences=sentences,
    vector_size=50,
    window=5,
    min_count=1,
    epochs=100,
    seed=42
)

print(f"FastText vocabulary  : {len(ft_model.wv)}")
print(f"Vector dimensions    : {ft_model.vector_size}")

# FastText handles out of vocabulary words
oov_words = ["learningg", "machinelearning", "deeplearningg", "pythoon"]

print(f"\nOut of Vocabulary Word Handling:")
print(f"{'Word':<20} {'In Vocab':<12} {'Has Vector'}")
print("-" * 45)
for word in oov_words:
    in_vocab   = word in ft_model.wv.key_to_index
    has_vector = True
    try:
        vec = ft_model.wv[word]
    except Exception:
        has_vector = False
    print(f"{word:<20} {str(in_vocab):<12} {has_vector}")

print("\nFastText can give vectors even for misspelled words!")
print("Word2Vec would fail completely for these words")

# Similarity with FastText
print(f"\nFastText Similarities:")
ft_pairs = [
    ("king",    "queen"),
    ("machine", "learning"),
    ("deep",    "neural"),
    ("cat",     "dog"),
]
for w1, w2 in ft_pairs:
    try:
        sim = ft_model.wv.similarity(w1, w2)
        print(f"  {w1:<15} vs {w2:<15} : {sim:.4f}")
    except Exception as e:
        print(f"  Error: {e}")


# ------------------------------------------
# PART 8 - COMPARISON
# ------------------------------------------

print("\n===== PART 8: Comparison =====")

print("""
COMPARISON TABLE:

Method       Similarity  OOV Words  Speed   Size
----------   ----------  ---------  -----   ----
One Hot      None        No         Fast    Huge
TF-IDF       Low         No         Fast    Large
Word2Vec     Good        No         Fast    Medium
GloVe        Good        No         Fast    Medium
FastText     Good        Yes        Medium  Large
BERT         Excellent   Yes        Slow    Huge

WHICH TO USE:
    Simple classification  -> TF-IDF
    Word similarity tasks  -> Word2Vec or GloVe
    Handling typos         -> FastText
    State of the art NLP   -> BERT or GPT embeddings
    We cover BERT in Day 27!
""")


# ------------------------------------------
# MINI PROJECT - Document Search Engine
# ------------------------------------------

print("===== MINI PROJECT: Document Search Engine =====")

print("""
Building a simple search engine using Word2Vec
Each document represented as average of word vectors
Query matched to most similar document
""")

# Documents
documents = [
    "machine learning is used for data analysis and prediction",
    "deep learning neural networks process images and text",
    "natural language processing understands human language",
    "python programming is popular for data science",
    "transformers and attention mechanism revolutionized nlp",
    "convolutional neural networks are used for image recognition",
    "word embeddings represent words as dense vectors",
    "reinforcement learning trains agents to make decisions",
    "data preprocessing is important step in machine learning",
    "bert and gpt are powerful language models",
]

def get_document_vector(doc, model):
    words  = doc.lower().split()
    vecs   = []
    for word in words:
        if word in model.wv:
            vecs.append(model.wv[word])
    if len(vecs) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vecs, axis=0)

def get_query_vector(query, model):
    return get_document_vector(query, model)

def search(query, documents, model, top_k=3):
    query_vec = get_query_vector(query, model)
    doc_vecs  = [get_document_vector(doc, model)
                 for doc in documents]

    similarities = []
    for i, doc_vec in enumerate(doc_vecs):
        if np.any(doc_vec) and np.any(query_vec):
            sim = cosine_similarity(
                [query_vec], [doc_vec])[0][0]
        else:
            sim = 0.0
        similarities.append((i, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

# Train on documents
doc_sentences = [doc.split() for doc in documents]
search_model  = Word2Vec(
    sentences=doc_sentences,
    vector_size=50,
    window=3,
    min_count=1,
    epochs=100,
    seed=42
)

# Search queries
queries = [
    "how do neural networks work",
    "python for data analysis",
    "language model for text",
    "image classification deep learning",
]

for query in queries:
    print(f"\nQuery: '{query}'")
    results = search(query, documents, search_model)
    print(f"Top Results:")
    for rank, (idx, score) in enumerate(results, 1):
        print(f"  {rank}. [{score:.3f}] {documents[idx]}")


print("\n===== WHAT I LEARNED TODAY =====")
print("Problem with one hot - no similarity")
print("Word2Vec - learn word vectors from context")
print("Word similarity - cosine similarity on vectors")
print("Word arithmetic - king - man + woman = queen")
print("GloVe - global statistics based embeddings")
print("FastText - handles out of vocabulary words")
print("Comparison - when to use which")
print("Mini Project - Document Search Engine")
print("\nDay 23 Done! Tomorrow - RNNs and LSTMs!")
