#======================================================
# DAY 22 - NLP Basics
# Tokenization, stemming, Stipwords, TF-IDF
# Author: Prateek kumar kuntal
# Date: 26 May 2025
#======================================================

import nltk
import string
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import (
    CountVectorizer, TfidfVectorizer)

# Download required NLTK data
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger")


# ------------------------------------------
# PART 1 - WHAT IS NLP
# ------------------------------------------

print("===== PART 1: What is NLP =====")

print("""
NLP (Natural Language Processing):
    Teaching computers to understand
    and generate human language

    Humans communicate in text and speech
    Computers understand only numbers
    NLP bridges this gap

REAL WORLD APPLICATIONS:
    ChatGPT, Claude, Gemini  - conversation
    Google Translate          - translation
    Gmail Smart Reply         - email suggestions
    Siri, Alexa               - voice assistants
    Grammarly                 - grammar correction
    Spam filters              - email classification
    Sentiment Analysis        - product reviews
    Search engines            - query understanding

NLP PIPELINE:
    Raw Text
        Tokenization
        Cleaning (lowercase, punctuation)
        Stopword Removal
        Stemming or Lemmatization
        Feature Extraction (TF-IDF, embeddings)
        Model (classification, generation)
        Output
""")


# ------------------------------------------
# PART 2 - TOKENIZATION
# ------------------------------------------

print("===== PART 2: Tokenization =====")

print("""
TOKENIZATION:
    Split text into smaller units called tokens
    Tokens can be words, sentences, subwords

    This is always the first step in NLP
    Every LLM tokenizes input before processing

TYPES:
    Word Tokenization    - split into words
    Sentence Tokenization- split into sentences
    Subword Tokenization - split into subwords
                           used by BERT, GPT
""")

text = """
Natural Language Processing is a fascinating field.
It helps computers understand human language.
Applications include chatbots, translation, and sentiment analysis.
Prateek is learning NLP to build GenAI applications.
"""

print("Original Text:")
print(text)

# Word tokenization
word_tokens = word_tokenize(text)
print(f"Word Tokens ({len(word_tokens)} tokens):")
print(word_tokens)

# Sentence tokenization
sent_tokens = sent_tokenize(text)
print(f"\nSentence Tokens ({len(sent_tokens)} sentences):")
for i, sent in enumerate(sent_tokens):
    print(f"  Sentence {i+1}: {sent.strip()}")

# Character tokenization
char_tokens = list(text[:50])
print(f"\nCharacter Tokens (first 50 chars):")
print(char_tokens)

# Simple word tokenization without NLTK
simple_tokens = text.lower().split()
print(f"\nSimple Split Tokens ({len(simple_tokens)}):")
print(simple_tokens[:15])


# ------------------------------------------
# PART 3 - TEXT CLEANING
# ------------------------------------------

print("\n===== PART 3: Text Cleaning =====")

print("""
TEXT CLEANING:
    Raw text is messy and inconsistent
    Need to standardize before processing

COMMON CLEANING STEPS:
    Lowercase         - "Hello" and "hello" same word
    Remove punctuation- "word." and "word" same token
    Remove numbers    - often not useful
    Remove extra spaces
    Remove special characters
""")

sample = "Hello World!!! This is NLP 101. Ready to learn? Let's GO!!!"
print(f"Original  : {sample}")

# Lowercase
lowered = sample.lower()
print(f"Lowercase : {lowered}")

# Remove punctuation
no_punct = lowered.translate(
    str.maketrans("", "", string.punctuation))
print(f"No Punct  : {no_punct}")

# Remove extra spaces
cleaned = " ".join(no_punct.split())
print(f"Cleaned   : {cleaned}")

# Remove numbers
import re
no_numbers = re.sub(r"\d+", "", cleaned)
print(f"No Numbers: {no_numbers}")

# Full cleaning function
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = " ".join(text.split())
    return text

test_texts = [
    "Hello World! This is Amazing123.",
    "Python3 is GREAT for NLP!!!",
    "I love   Machine   Learning...",
]

print("\nCleaning multiple texts:")
for t in test_texts:
    print(f"  Before: {t}")
    print(f"  After : {clean_text(t)}")
    print()


# ------------------------------------------
# PART 4 - STOPWORDS
# ------------------------------------------

print("===== PART 4: Stopwords =====")

print("""
STOPWORDS:
    Common words that carry little meaning
    Examples: the, is, are, and, but, in, on

    Removing them reduces noise
    Keeps only meaningful content words

    But sometimes stopwords matter:
    "not good" vs "good" - removing "not" changes meaning
    Context determines whether to remove them
""")

stop_words = set(stopwords.words("english"))
print(f"Total English stopwords : {len(stop_words)}")
print(f"Sample stopwords        : {list(stop_words)[:20]}")

text_sample = "This is a very good movie and I really enjoyed watching it"
tokens      = word_tokenize(text_sample.lower())

filtered = [word for word in tokens
            if word not in stop_words
            and word not in string.punctuation]

print(f"\nOriginal  : {text_sample}")
print(f"Tokens    : {tokens}")
print(f"Filtered  : {filtered}")
print(f"\nBefore    : {len(tokens)} words")
print(f"After     : {len(filtered)} words")
print(f"Removed   : {len(tokens) - len(filtered)} stopwords")


# ------------------------------------------
# PART 5 - STEMMING
# ------------------------------------------

print("\n===== PART 5: Stemming =====")

print("""
STEMMING:
    Reduce words to their root/base form
    Cuts off suffixes using rules

    running   -> run
    played    -> play
    studies   -> studi  (not perfect!)
    better    -> better (misses this)

    Fast but sometimes inaccurate
    Produces stems not real words
    Used when speed matters more than accuracy
""")

stemmer = PorterStemmer()

words_to_stem = [
    "running", "runs", "runner", "ran",
    "playing", "played", "plays", "player",
    "studying", "studies", "studied", "student",
    "learning", "learned", "learner",
    "better", "best", "good",
    "happiness", "happy", "happily",
]

print(f"{'Word':<15} {'Stem'}")
print("-" * 30)
for word in words_to_stem:
    stem = stemmer.stem(word)
    print(f"{word:<15} {stem}")


# ------------------------------------------
# PART 6 - LEMMATIZATION
# ------------------------------------------

print("\n===== PART 6: Lemmatization =====")

print("""
LEMMATIZATION:
    Reduce words to their dictionary base form
    Uses vocabulary and grammar rules
    Produces real words unlike stemming

    running   -> run
    better    -> good   (understands context)
    studies   -> study  (correct!)
    was       -> be

    Slower than stemming but more accurate
    Preferred for most NLP tasks today
""")

lemmatizer = WordNetLemmatizer()

words_to_lemm = [
    "running", "runs", "runner",
    "playing", "played", "plays",
    "studying", "studies", "studied",
    "better", "best",
    "was", "were", "been",
    "happily", "happiness",
    "mice", "geese", "children",
]

print(f"{'Word':<15} {'Stem':<15} {'Lemma'}")
print("-" * 45)
for word in words_to_lemm:
    stem  = stemmer.stem(word)
    lemma = lemmatizer.lemmatize(word)
    print(f"{word:<15} {stem:<15} {lemma}")

print("""
WHEN TO USE WHICH:
    Stemming      - fast processing, search engines
    Lemmatization - accuracy matters, sentiment analysis
    Neither       - deep learning models (BERT, GPT)
                    they handle this internally
""")


# ------------------------------------------
# PART 7 - BAG OF WORDS
# ------------------------------------------

print("===== PART 7: Bag of Words =====")

print("""
BAG OF WORDS (BOW):
    Represent text as word frequency counts
    Ignore word order and grammar
    Just count how many times each word appears

    Documents become vectors of word counts
    Same length for all documents
    Length = size of vocabulary

EXAMPLE:
    Vocab     : [cat, dog, sat, mat, the]
    "the cat sat" -> [1, 0, 1, 0, 1]
    "the dog sat" -> [0, 1, 1, 0, 1]

PROBLEM:
    Common words get high counts
    but carry little meaning
    TF-IDF solves this problem
""")

documents = [
    "I love machine learning and deep learning",
    "Deep learning is a subset of machine learning",
    "Natural language processing uses machine learning",
    "I enjoy learning new things about artificial intelligence",
    "Machine learning models learn from data",
]

vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform(documents)

vocab = vectorizer.get_feature_names_out()
print(f"Vocabulary ({len(vocab)} words):")
print(vocab)

print(f"\nBag of Words Matrix shape: {bow_matrix.shape}")
print("(5 documents x vocabulary size)")

bow_array = bow_matrix.toarray()
print(f"\nDocument 1: {documents[0]}")
print(f"BOW vector: {bow_array[0]}")

print(f"\nDocument 2: {documents[1]}")
print(f"BOW vector: {bow_array[1]}")


# ------------------------------------------
# PART 8 - TF-IDF
# ------------------------------------------

print("\n===== PART 8: TF-IDF =====")

print("""
TF-IDF (Term Frequency - Inverse Document Frequency):
    Better than simple word counts
    Gives high score to:
        Words that appear often in ONE document
        but rarely across ALL documents

TF (Term Frequency):
    How often word appears in document
    TF = count of word / total words in document

IDF (Inverse Document Frequency):
    How rare the word is across all documents
    IDF = log(total docs / docs containing word)
    Common words (the, is) get low IDF
    Rare words get high IDF

TF-IDF = TF * IDF
    High score = word is important for this document
    Low score  = word is common everywhere (not useful)

USED IN:
    Search engines (Google)
    Document similarity
    Keyword extraction
    Text classification
""")

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix     = tfidf_vectorizer.fit_transform(documents)

tfidf_vocab = tfidf_vectorizer.get_feature_names_out()
tfidf_array = tfidf_matrix.toarray()

print(f"TF-IDF Matrix shape: {tfidf_array.shape}")

print(f"\nDocument 1: {documents[0]}")
print(f"TF-IDF scores:")
for word, score in zip(tfidf_vocab, tfidf_array[0]):
    if score > 0:
        print(f"  {word:<20} : {score:.4f}")

print(f"\nDocument 2: {documents[1]}")
print(f"TF-IDF scores:")
for word, score in zip(tfidf_vocab, tfidf_array[1]):
    if score > 0:
        print(f"  {word:<20} : {score:.4f}")

# Document similarity using TF-IDF
print("\nDocument Similarity using TF-IDF:")
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(tfidf_matrix)

print(f"{'Doc':<8}", end="")
for i in range(len(documents)):
    print(f"{'Doc'+str(i+1):<10}", end="")
print()
print("-" * 58)

for i in range(len(documents)):
    print(f"Doc{i+1:<5}", end="")
    for j in range(len(documents)):
        print(f"{similarity_matrix[i][j]:<10.3f}", end="")
    print()

# Find most similar document to Doc 1
doc1_sims  = similarity_matrix[0].copy()
doc1_sims[0] = 0  # exclude itself
most_similar = np.argmax(doc1_sims) + 1
print(f"\nMost similar to Doc1: Doc{most_similar}")
print(f"Doc1: {documents[0]}")
print(f"Doc{most_similar}: {documents[most_similar-1]}")


# ------------------------------------------
# MINI PROJECT - Text Classification with TF-IDF
# ------------------------------------------

print("\n===== MINI PROJECT: Spam Classifier with TF-IDF =====")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Dataset
spam_emails = [
    "Win a free iPhone now click here",
    "Congratulations you won 1 million dollars",
    "Free money transfer click this link",
    "Buy cheap medicine online no prescription",
    "You are selected for lottery prize claim now",
    "Make money fast from home guaranteed",
    "Click here to claim your free gift",
    "Urgent your account will be closed verify now",
    "Hot singles in your area click now",
    "Free casino bonus limited time offer",
    "Investment opportunity double your money",
    "Lose weight fast with this miracle pill",
]

normal_emails = [
    "Hi Prateek can we schedule a meeting tomorrow",
    "Please find attached the project report",
    "The team lunch is at 1pm today",
    "Can you review my pull request when free",
    "Happy birthday hope you have a great day",
    "The quarterly results meeting is on Friday",
    "Thanks for your help with the presentation",
    "Please submit your timesheet by end of day",
    "The new feature deployment is scheduled tonight",
    "Can we discuss the project timeline next week",
    "Great work on the machine learning project",
    "The client meeting went really well today",
]

texts  = spam_emails + normal_emails
labels = [1] * len(spam_emails) + [0] * len(normal_emails)

print(f"Total emails     : {len(texts)}")
print(f"Spam emails      : {sum(labels)}")
print(f"Normal emails    : {len(labels) - sum(labels)}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.25, random_state=42)

# TF-IDF features
tfidf = TfidfVectorizer(
    max_features=100,
    ngram_range=(1, 2)   # unigrams and bigrams
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

# Train classifier
clf = LogisticRegression(random_state=42)
clf.fit(X_train_tfidf, y_train)

# Evaluate
y_pred   = clf.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy   : {accuracy*100:.2f}%")
print(f"\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["Normal", "Spam"]))

# Test new emails
new_emails = [
    "Click here to win a free prize now",
    "Hi can we meet for coffee tomorrow",
    "Congratulations you have been selected",
    "Please review the attached document",
    "Free money guaranteed limited time",
]

new_tfidf = tfidf.transform(new_emails)
new_preds = clf.predict(new_tfidf)
new_probs = clf.predict_proba(new_tfidf)

print("Predictions on New Emails:")
print(f"{'Email':<45} {'Label':<10} {'Confidence'}")
print("-" * 65)
for email, pred, prob in zip(new_emails, new_preds, new_probs):
    label      = "SPAM" if pred == 1 else "Normal"
    confidence = max(prob) * 100
    print(f"{email[:43]:<45} {label:<10} {confidence:.1f}%")

# Top spam keywords
feature_names = tfidf.get_feature_names_out()
coefs         = clf.coef_[0]

top_spam_idx   = coefs.argsort()[-10:][::-1]
top_normal_idx = coefs.argsort()[:10]

print(f"\nTop Spam Keywords:")
for idx in top_spam_idx:
    print(f"  {feature_names[idx]:<20} score: {coefs[idx]:.4f}")

print(f"\nTop Normal Keywords:")
for idx in top_normal_idx:
    print(f"  {feature_names[idx]:<20} score: {coefs[idx]:.4f}")


print("\n===== WHAT I LEARNED TODAY =====")
print("NLP Pipeline - text to features to model")
print("Tokenization - word, sentence, character")
print("Text Cleaning - lowercase, punctuation, regex")
print("Stopwords - remove common meaningless words")
print("Stemming - fast but inaccurate root finding")
print("Lemmatization - accurate dictionary base form")
print("Bag of Words - word count vectors")
print("TF-IDF - smart word importance scoring")
print("Mini Project - Spam Classifier with TF-IDF")
print("\nDay 22 Done! Tomorrow - Word Embeddings!")