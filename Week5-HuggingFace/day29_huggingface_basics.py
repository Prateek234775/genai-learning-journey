# ============================================
# DAY 29 - HuggingFace Basics
# Pipelines, Tokenizers, Model Hub
# Author: Prateek Kumar Kuntal
# Date: 02 June 2026
# ============================================

from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)
import torch
import numpy as np


# ------------------------------------------
# PART 1 - WHAT IS HUGGINGFACE
# ------------------------------------------

print("===== PART 1: What is HuggingFace =====")

print("""
HUGGINGFACE:
    The GitHub of Machine Learning
    Platform for sharing models, datasets, spaces
    Open source library for NLP and beyond

    Founded in 2016, now valued at 4.5 billion dollars
    Used by Google, Microsoft, Amazon, Meta
    Over 500,000 models available for free

HUGGINGFACE ECOSYSTEM:
    Transformers  - model library (what we use today)
    Datasets      - dataset library
    Tokenizers    - fast tokenization
    Accelerate    - distributed training
    PEFT          - parameter efficient fine tuning
    Diffusers     - image generation models
    Gradio        - build AI demos quickly
    Spaces        - host AI apps for free

WHY HUGGINGFACE MATTERS FOR YOU:
    No need to build models from scratch
    Use state of the art pretrained models
    Fine tune on your own data easily
    Deploy models for free on Spaces
    This is what companies use in production

KEY CONCEPTS:
    Model Hub    - repository of pretrained models
    Pipeline     - simple one line inference
    Tokenizer    - converts text to tokens
    AutoClass    - automatically loads right model
    Checkpoint   - saved model weights
""")


# ------------------------------------------
# PART 2 - HUGGINGFACE PIPELINES
# ------------------------------------------

print("===== PART 2: HuggingFace Pipelines =====")

print("""
PIPELINE:
    Simplest way to use pretrained models
    One line of code for inference
    Handles tokenization and post processing

    pipeline("task", model="model-name")

AVAILABLE TASKS:
    text-classification  - sentiment, topic
    text-generation      - GPT style generation
    fill-mask            - BERT style masking
    question-answering   - extractive QA
    summarization        - text summarization
    translation          - language translation
    zero-shot-classification - classify without training
    ner                  - named entity recognition
    feature-extraction   - get embeddings
""")

# Task 1 - Sentiment Analysis
print("--- Task 1: Sentiment Analysis ---")
print("Loading sentiment analysis pipeline...")

sentiment_pipeline = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1    # -1 for CPU, 0 for GPU
)

texts = [
    "I absolutely love this product it is amazing",
    "This is the worst thing I have ever bought",
    "The weather today is okay nothing special",
    "HuggingFace makes machine learning so easy",
    "I am very disappointed with the service",
]

print(f"\n{'Text':<50} {'Label':<12} {'Score'}")
print("-" * 70)
for text in texts:
    result = sentiment_pipeline(text)[0]
    print(f"{text[:48]:<50} {result['label']:<12} {result['score']:.4f}")


# Task 2 - Zero Shot Classification
print("\n--- Task 2: Zero Shot Classification ---")
print("Loading zero shot pipeline...")

zero_shot = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=-1
)

text       = "The new iPhone has an amazing camera and great battery life"
categories = ["technology", "sports", "food", "politics", "entertainment"]

result = zero_shot(text, candidate_labels=categories)

print(f"\nText: {text}")
print(f"\nClassification Results:")
for label, score in zip(result["labels"], result["scores"]):
    bar = "#" * int(score * 30)
    print(f"  {label:<15} : {bar:<30} {score:.4f}")


# Task 3 - Named Entity Recognition
print("\n--- Task 3: Named Entity Recognition ---")
print("Loading NER pipeline...")

ner_pipeline = pipeline(
    "ner",
    model="dbmdz/bert-large-cased-finetuned-conll03-english",
    aggregation_strategy="simple",
    device=-1
)

text = "Prateek Kumar studies at VIT Bhopal in India and wants to work at Google or Microsoft"

entities = ner_pipeline(text)
print(f"\nText: {text}")
print(f"\nEntities Found:")
print(f"{'Entity':<25} {'Type':<12} {'Score'}")
print("-" * 45)
for entity in entities:
    print(f"{entity['word']:<25} {entity['entity_group']:<12} {entity['score']:.4f}")


# Task 4 - Question Answering
print("\n--- Task 4: Question Answering ---")
print("Loading QA pipeline...")

qa_pipeline = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad",
    device=-1
)

context = """
HuggingFace is an AI company founded in 2016. It provides open source
tools for machine learning including the Transformers library which has
over 500000 pretrained models. The company is headquartered in New York
and Paris. HuggingFace raised 235 million dollars in funding in 2022
at a valuation of 4.5 billion dollars. Their platform is used by over
10000 companies worldwide including Google Microsoft and Amazon.
"""

questions = [
    "When was HuggingFace founded?",
    "How many pretrained models does HuggingFace have?",
    "Where is HuggingFace headquartered?",
    "How much funding did HuggingFace raise?",
]

print(f"\nContext: {context[:100]}...")
print(f"\nQuestion Answering:")
print(f"{'Question':<45} {'Answer':<30} {'Score'}")
print("-" * 80)
for question in questions:
    result = qa_pipeline(question=question, context=context)
    print(f"{question:<45} {result['answer']:<30} {result['score']:.4f}")


# Task 5 - Text Summarization
print("\n--- Task 5: Text Summarization ---")
print("Loading summarization pipeline...")

summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=-1
)

long_text = """
Machine learning is a subset of artificial intelligence that provides
systems the ability to automatically learn and improve from experience
without being explicitly programmed. Machine learning focuses on the
development of computer programs that can access data and use it to
learn for themselves. The process begins with observations or data such
as examples, direct experience, or instruction, so that computers can
learn to make better decisions in the future. The primary aim is to
allow computers to learn automatically without human intervention or
assistance and adjust actions accordingly. Machine learning algorithms
are used in a wide variety of applications such as email filtering and
computer vision where it is difficult or infeasible to develop
conventional algorithms to perform the needed tasks.
"""

summary = summarizer(
    long_text,
    max_length=80,
    min_length=30,
    do_sample=False
)

print(f"\nOriginal text length : {len(long_text.split())} words")
print(f"Summary length       : {len(summary[0]['summary_text'].split())} words")
print(f"\nSummary:")
print(summary[0]["summary_text"])


# Task 6 - Fill Mask (BERT style)
print("\n--- Task 6: Fill Mask ---")
print("Loading fill mask pipeline...")

fill_mask = pipeline(
    "fill-mask",
    model="bert-base-uncased",
    device=-1
)

masked_sentences = [
    "Machine learning is a [MASK] of artificial intelligence.",
    "Python is the most popular [MASK] for data science.",
    "The capital of France is [MASK].",
    "HuggingFace is the [MASK] of machine learning.",
]

for sentence in masked_sentences:
    results = fill_mask(sentence)
    print(f"\nMasked: {sentence}")
    print(f"Top predictions:")
    for r in results[:3]:
        print(f"  {r['token_str']:<20} score: {r['score']:.4f}")


# ------------------------------------------
# PART 3 - TOKENIZERS
# ------------------------------------------

print("\n===== PART 3: Tokenizers =====")

print("""
TOKENIZER:
    Converts raw text to token ids
    Handles special tokens automatically
    Manages padding and truncation
    Each model has its own tokenizer

TOKENIZER STEPS:
    1. Text normalization (lowercase etc)
    2. Pre-tokenization (split on spaces)
    3. Tokenization (apply vocab)
    4. Post processing (add special tokens)

DIFFERENT TOKENIZERS:
    WordPiece  - BERT
    BPE        - GPT2, RoBERTa
    SentencePiece - T5, ALBERT
    Unigram    - XLNet
""")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased")

text = "HuggingFace transformers library is amazing for NLP!"

print(f"Original text    : {text}")

# Tokenize
tokens = tokenizer.tokenize(text)
print(f"Tokens           : {tokens}")
print(f"Num tokens       : {len(tokens)}")

# Encode
encoding = tokenizer(
    text,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=64
)
print(f"\nInput IDs        : {encoding['input_ids']}")
print(f"Attention Mask   : {encoding['attention_mask']}")
print(f"Input IDs shape  : {encoding['input_ids'].shape}")

# Decode back
decoded = tokenizer.decode(
    encoding["input_ids"][0],
    skip_special_tokens=True
)
print(f"\nDecoded back     : {decoded}")

# Special tokens
print(f"\nSpecial tokens:")
print(f"  CLS token    : {tokenizer.cls_token} "
      f"(id: {tokenizer.cls_token_id})")
print(f"  SEP token    : {tokenizer.sep_token} "
      f"(id: {tokenizer.sep_token_id})")
print(f"  PAD token    : {tokenizer.pad_token} "
      f"(id: {tokenizer.pad_token_id})")
print(f"  MASK token   : {tokenizer.mask_token} "
      f"(id: {tokenizer.mask_token_id})")
print(f"  Vocab size   : {tokenizer.vocab_size}")

# Batch tokenization
texts = [
    "Short sentence.",
    "This is a much longer sentence with many more words in it.",
    "Medium length sentence here.",
]

batch = tokenizer(
    texts,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=32
)

print(f"\nBatch Tokenization:")
print(f"Input shape      : {batch['input_ids'].shape}")
print(f"(3 texts padded to same length)")
print(f"\nInput IDs:")
print(batch["input_ids"])
print(f"\nAttention Mask (1=real, 0=padding):")
print(batch["attention_mask"])


# ------------------------------------------
# PART 4 - AUTOMODEL AND EMBEDDINGS
# ------------------------------------------

print("\n===== PART 4: AutoModel and Embeddings =====")

print("""
AUTOMODEL:
    Automatically loads correct model class
    No need to know exact model architecture
    Just provide model name from Hub

AUTOCLASS OPTIONS:
    AutoModel                       - base model
    AutoModelForSequenceClassification
    AutoModelForTokenClassification
    AutoModelForQuestionAnswering
    AutoModelForCausalLM            - text generation
    AutoModelForMaskedLM            - fill mask

GETTING EMBEDDINGS:
    Pass text through model
    Get hidden states from last layer
    Average to get sentence embedding
    Use for similarity, clustering, search
""")

print("Loading BERT base model for embeddings...")

tokenizer_bert = AutoTokenizer.from_pretrained(
    "bert-base-uncased")
model_bert     = AutoModel.from_pretrained(
    "bert-base-uncased")

def get_embeddings(texts, tokenizer, model):
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**encoded)

    # Mean pooling of last hidden state
    hidden_states   = outputs.last_hidden_state
    attention_mask  = encoded["attention_mask"]
    mask_expanded   = attention_mask.unsqueeze(-1).float()
    sum_embeddings  = (hidden_states * mask_expanded).sum(1)
    sum_mask        = mask_expanded.sum(1).clamp(min=1e-9)
    embeddings      = sum_embeddings / sum_mask

    return embeddings.numpy()

sentences = [
    "I love machine learning",
    "I enjoy deep learning",
    "The weather is beautiful today",
    "It is a sunny and warm day",
    "Python is great for data science",
]

print("Computing BERT embeddings...")
embeddings = get_embeddings(sentences, tokenizer_bert, model_bert)

print(f"\nEmbedding shape  : {embeddings.shape}")
print(f"(5 sentences x 768 dimensions)")

# Compute similarity
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(embeddings)

print(f"\nSentence Similarity Matrix:")
print(f"{'':>5}", end="")
for i in range(len(sentences)):
    print(f"S{i+1:>6}", end="")
print()

for i in range(len(sentences)):
    print(f"S{i+1:>4}", end="")
    for j in range(len(sentences)):
        print(f"{similarity_matrix[i][j]:>7.3f}", end="")
    print()

print(f"\nMost similar pairs:")
pairs = []
for i in range(len(sentences)):
    for j in range(i+1, len(sentences)):
        pairs.append((i, j, similarity_matrix[i][j]))

pairs.sort(key=lambda x: x[2], reverse=True)
for i, j, score in pairs[:3]:
    print(f"  S{i+1} and S{j+1}: {score:.4f}")
    print(f"    '{sentences[i]}'")
    print(f"    '{sentences[j]}'")


# ------------------------------------------
# PART 5 - MODEL HUB
# ------------------------------------------

print("\n===== PART 5: Model Hub =====")

print("""
HUGGINGFACE MODEL HUB:
    huggingface.co/models
    500000+ models available
    Filter by task, language, library, size

HOW TO FIND RIGHT MODEL:
    1. Go to huggingface.co/models
    2. Filter by task (text-classification etc)
    3. Sort by downloads or likes
    4. Check model card for details
    5. Copy model name and use in code

POPULAR MODELS TO KNOW:
    bert-base-uncased           - BERT base English
    distilbert-base-uncased     - smaller faster BERT
    roberta-base                - improved BERT
    gpt2                        - GPT-2 small
    facebook/bart-large-cnn     - summarization
    Helsinki-NLP/opus-mt-en-hi  - English to Hindi
    sentence-transformers/...   - sentence embeddings

MODEL CARD:
    Description of model
    Training data and methodology
    Performance metrics
    Usage examples
    Limitations and biases

ALWAYS CHECK:
    Number of downloads (popularity)
    Last updated date (maintenance)
    License (for commercial use)
    Model size (fits in your memory)

UPLOADING YOUR OWN MODEL:
    pip install huggingface_hub
    huggingface-cli login
    model.push_to_hub("your-model-name")
    Visible to everyone on Hub!
""")

# Show how to explore model info
print("Example model information lookup:")

from transformers import AutoConfig

model_names = [
    "bert-base-uncased",
    "distilbert-base-uncased",
    "gpt2",
]

for model_name in model_names:
    config = AutoConfig.from_pretrained(model_name)
    print(f"\nModel: {model_name}")
    print(f"  Architecture : {config.model_type}")
    if hasattr(config, "hidden_size"):
        print(f"  Hidden size  : {config.hidden_size}")
    if hasattr(config, "num_hidden_layers"):
        print(f"  Layers       : {config.num_hidden_layers}")
    if hasattr(config, "num_attention_heads"):
        print(f"  Attn heads   : {config.num_attention_heads}")
    if hasattr(config, "vocab_size"):
        print(f"  Vocab size   : {config.vocab_size}")


# ------------------------------------------
# MINI PROJECT - Multi Task NLP System
# ------------------------------------------

print("\n===== MINI PROJECT: Multi Task NLP System =====")

print("""
Building a complete NLP system that:
    1. Classifies sentiment of reviews
    2. Extracts named entities
    3. Answers questions about a document
    4. Summarizes long text
All using HuggingFace pipelines
""")

# Product review analysis system
reviews = [
    "Apple released an amazing new MacBook Pro with M3 chip in San Francisco. The performance is incredible and battery life is outstanding.",
    "The Samsung Galaxy S24 launched in Seoul has terrible battery issues. Users in New York and London are very disappointed with the product.",
    "Google unveiled Gemini AI in Mountain View California. Sundar Pichai says it will revolutionize search and help millions of users worldwide.",
]

print("Analyzing product reviews with multiple NLP tasks...")
print("=" * 70)

for i, review in enumerate(reviews):
    print(f"\nReview {i+1}: {review[:60]}...")

    # Sentiment
    sentiment = sentiment_pipeline(review)[0]
    print(f"\n  Sentiment: {sentiment['label']} "
          f"(confidence: {sentiment['score']:.3f})")

    # Entities
    entities = ner_pipeline(review)
    if entities:
        print(f"  Entities found:")
        for ent in entities:
            print(f"    {ent['word']:<20} [{ent['entity_group']}]")

    print("-" * 70)

# Question answering on a tech document
tech_context = """
HuggingFace was founded by Clement Delangue, Julien Chaumond and
Thomas Wolf in 2016. The company started as a chatbot app for teenagers
but pivoted to focus on natural language processing tools. The
Transformers library was released in 2019 and quickly became the
most popular open source library for machine learning. As of 2024
HuggingFace has over 500000 models on its platform and is used by
more than 10000 companies. The company raised 235 million dollars
in Series D funding in 2022 and is headquartered in New York City.
"""

tech_questions = [
    "Who founded HuggingFace?",
    "When was the Transformers library released?",
    "How much did HuggingFace raise in Series D?",
]

print(f"\nDocument QA System:")
print(f"Context: {tech_context[:80]}...")
print()
for question in tech_questions:
    answer = qa_pipeline(
        question=question, context=tech_context)
    print(f"Q: {question}")
    print(f"A: {answer['answer']} "
          f"(confidence: {answer['score']:.3f})")
    print()


print("\n===== WHAT I LEARNED TODAY =====")
print("HuggingFace ecosystem and its importance")
print("Pipelines - one line inference for any task")
print("Sentiment analysis with DistilBERT")
print("Zero shot classification without training")
print("Named entity recognition")
print("Question answering with context")
print("Text summarization with BART")
print("Fill mask with BERT")
print("Tokenizers - text to tokens and back")
print("AutoModel - get embeddings from BERT")
print("Model Hub - finding right pretrained model")
print("Mini Project - Multi task NLP system")
print("\nDay 29 Done! Tomorrow - Text Classification with BERT!")