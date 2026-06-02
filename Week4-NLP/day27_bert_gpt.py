# ============================================
# DAY 27 - BERT and GPT
# How They Differ, What They Do
# Author: Prateek Kumar Kuntal
# Date: 31 May 2025
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# ------------------------------------------
# PART 1 - BERT vs GPT OVERVIEW
# ------------------------------------------

print("===== PART 1: BERT vs GPT Overview =====")

print("""
BERT (Bidirectional Encoder Representations from Transformers):
    Released by Google in 2018
    Encoder only transformer
    Bidirectional - reads entire sequence at once
    Pre-trained on two tasks:
        Masked Language Modeling (MLM)
        Next Sentence Prediction (NSP)

    BERT reads:  "The [MASK] sat on the mat"
    Predicts:    "cat" using both left and right context

    Good for UNDERSTANDING tasks:
        Text classification
        Named entity recognition
        Question answering
        Sentence similarity

GPT (Generative Pre-trained Transformer):
    Released by OpenAI in 2018
    Decoder only transformer
    Unidirectional - reads left to right only
    Pre-trained on language modeling:
        Predict next token given previous tokens

    GPT reads:   "The cat sat on"
    Predicts:    "the" then "mat" then "." etc.

    Good for GENERATION tasks:
        Text generation
        Story writing
        Code generation
        Conversation

KEY DIFFERENCE:
    BERT  - sees full context  - better understanding
    GPT   - sees past only     - better generation
    Today GPT style dominates because generation
    scales better and can do understanding too
""")


# ------------------------------------------
# PART 2 - BERT PRE-TRAINING TASKS
# ------------------------------------------

print("===== PART 2: BERT Pre-training Tasks =====")

print("""
MASKED LANGUAGE MODELING (MLM):
    Randomly mask 15% of input tokens
    Model predicts masked tokens
    Forces model to understand context

    Input : "The [MASK] sat on the [MASK]"
    Output: "cat"              "mat"

    Why masking works:
    Model cannot just copy input
    Must understand surrounding words
    Learns rich bidirectional representations

NEXT SENTENCE PREDICTION (NSP):
    Given two sentences predict if B follows A

    Positive: "I love pizza." "It is my favorite food."
    Negative: "I love pizza." "The sky is blue."

    Helps model understand sentence relationships
    Useful for question answering and inference

BERT TOKENIZATION - WORDPIECE:
    Splits words into subwords
    "playing" -> ["play", "##ing"]
    "unbelievable" -> ["un", "##believ", "##able"]
    Handles rare and unknown words
    Vocabulary of 30000 subword tokens

SPECIAL TOKENS:
    [CLS] - classification token at start
    [SEP] - separator between sentences
    [MASK]- masked token to predict
    [PAD] - padding token
""")

# Simulate BERT tokenization
def simple_wordpiece_tokenize(text):
    # Simplified wordpiece tokenization
    tokens = []
    for word in text.lower().split():
        if len(word) <= 4:
            tokens.append(word)
        else:
            # Split into subwords
            tokens.append(word[:4])
            remaining = word[4:]
            while remaining:
                tokens.append("##" + remaining[:3])
                remaining = remaining[3:]
    return tokens

texts = [
    "playing football is amazing",
    "unbelievable performance today",
    "transformers revolutionized NLP",
]

print("Simplified WordPiece Tokenization:")
for text in texts:
    tokens = simple_wordpiece_tokenize(text)
    print(f"  Input  : {text}")
    print(f"  Tokens : {tokens}")
    print()

# Masked Language Modeling simulation
def create_mlm_input(tokens, mask_prob=0.15):
    masked_tokens = tokens.copy()
    labels        = [-100] * len(tokens)  # -100 = ignore

    for i, token in enumerate(tokens):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
        if np.random.random() < mask_prob:
            labels[i]        = i  # original token index
            masked_tokens[i] = "[MASK]"

    return masked_tokens, labels

np.random.seed(42)
example_tokens = ["[CLS]", "the", "cat", "sat",
                  "on", "the", "mat", "[SEP]"]
masked, labels = create_mlm_input(example_tokens)

print("Masked Language Modeling Example:")
print(f"Original : {example_tokens}")
print(f"Masked   : {masked}")
print(f"Labels   : {labels}")
print(f"(Label -100 means ignore, others are positions to predict)")


# ------------------------------------------
# PART 3 - BUILD BERT STYLE MODEL
# ------------------------------------------

print("\n===== PART 3: BERT Style Model =====")

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads

        self.W_q     = nn.Linear(d_model, d_model, bias=False)
        self.W_k     = nn.Linear(d_model, d_model, bias=False)
        self.W_v     = nn.Linear(d_model, d_model, bias=False)
        self.W_o     = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale   = self.d_k ** 0.5

    def split_heads(self, x):
        batch, seq, _ = x.shape
        x = x.view(batch, seq, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))

        scores  = (Q @ K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        weights = self.dropout(F.softmax(scores, dim=-1))
        output  = weights @ V
        output  = output.transpose(1, 2).contiguous()
        output  = output.view(output.shape[0], -1, self.d_model)
        return self.W_o(output), weights


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.attention    = MultiHeadAttention(
            d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),              # BERT uses GELU not ReLU
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, _ = self.attention(x, x, x, mask)
        x           = self.norm1(x + self.dropout(attn_out))
        ff_out      = self.feed_forward(x)
        x           = self.norm2(x + ff_out)
        return x


class BERTModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads,
                 d_ff, num_layers, max_len=512, dropout=0.1):
        super(BERTModel, self).__init__()

        # Three types of embeddings in BERT
        self.token_embedding    = nn.Embedding(
            vocab_size, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(
            max_len, d_model)
        self.segment_embedding  = nn.Embedding(
            3, d_model)     # segment A=1, B=2, PAD=0

        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers  = nn.ModuleList([
            TransformerEncoderLayer(
                d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.pooler  = nn.Linear(d_model, d_model)

    def forward(self, input_ids, segment_ids=None,
                attention_mask=None):
        batch, seq_len = input_ids.shape

        # Position ids
        pos_ids = torch.arange(
            seq_len, device=input_ids.device
        ).unsqueeze(0).expand(batch, -1)

        # Default segment ids
        if segment_ids is None:
            segment_ids = torch.ones_like(input_ids)

        # Combine embeddings
        x  = self.token_embedding(input_ids)
        x += self.position_embedding(pos_ids)
        x += self.segment_embedding(segment_ids)
        x  = self.dropout(self.norm(x))

        # Attention mask
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
        else:
            mask = None

        # Encoder layers
        for layer in self.layers:
            x = layer(x, mask)

        # CLS token representation for classification
        cls_output    = torch.tanh(self.pooler(x[:, 0, :]))

        return x, cls_output


class BERTForClassification(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads,
                 d_ff, num_layers, num_classes,
                 max_len=512, dropout=0.1):
        super(BERTForClassification, self).__init__()

        self.bert       = BERTModel(
            vocab_size, d_model, num_heads,
            d_ff, num_layers, max_len, dropout)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids, segment_ids=None,
                attention_mask=None):
        _, cls_out = self.bert(
            input_ids, segment_ids, attention_mask)
        cls_out    = self.dropout(cls_out)
        logits     = self.classifier(cls_out)
        return logits


# Test BERT model
bert = BERTForClassification(
    vocab_size  = 1000,
    d_model     = 64,
    num_heads   = 4,
    d_ff        = 128,
    num_layers  = 2,
    num_classes = 2
)

total_params = sum(p.numel() for p in bert.parameters())
print(f"BERT Style Model:")
print(f"Parameters       : {total_params:,}")

# Test forward pass
input_ids    = torch.randint(1, 1000, (2, 12))
segment_ids  = torch.ones(2, 12, dtype=torch.long)
attn_mask    = torch.ones(2, 12)

logits       = bert(input_ids, segment_ids, attn_mask)
print(f"Input shape      : {input_ids.shape}")
print(f"Output shape     : {logits.shape}")
print(f"(2 samples, 2 classes)")


# ------------------------------------------
# PART 4 - GPT PRE-TRAINING
# ------------------------------------------

print("\n===== PART 4: GPT Pre-training =====")

print("""
GPT PRE-TRAINING:
    Language modeling - predict next token
    Train on billions of tokens of text
    Each token learns to predict the next one

    Input : "The cat sat on the"
    Target: "cat sat on the mat"
    (Target is input shifted by one position)

    This is called CAUSAL language modeling
    Causal = can only see past not future
    Uses causal mask to prevent future peeking

GPT TOKENIZATION - BPE (Byte Pair Encoding):
    Starts with individual characters
    Merges most frequent pairs repeatedly
    "playing" -> ["play", "ing"]
    More efficient than WordPiece

GPT VERSIONS:
    GPT-1  : 117M parameters    2018
    GPT-2  : 1.5B parameters    2019
    GPT-3  : 175B parameters    2020
    GPT-4  : estimated 1T+      2023

SCALING LAW:
    More data + more parameters = better model
    Performance improves predictably with scale
    This insight led to the LLM revolution

ZERO SHOT AND FEW SHOT:
    GPT-3 showed models can follow instructions
    without any task specific fine tuning
    Just describe the task in the prompt!
    This was a massive breakthrough
""")

# GPT style language model
class GPTModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads,
                 d_ff, num_layers, max_len=512, dropout=0.1):
        super(GPTModel, self).__init__()

        self.token_embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=0)
        self.pos_embedding   = nn.Embedding(
            max_len, d_model)
        self.dropout         = nn.Dropout(dropout)
        self.norm            = nn.LayerNorm(d_model)

        self.layers          = nn.ModuleList([
            TransformerEncoderLayer(
                d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.lm_head         = nn.Linear(
            d_model, vocab_size, bias=False)

        # Weight tying - share embedding and output weights
        self.lm_head.weight  = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def create_causal_mask(self, seq_len, device):
        mask = torch.tril(
            torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, input_ids, targets=None):
        batch, seq_len = input_ids.shape
        device         = input_ids.device

        pos_ids = torch.arange(
            seq_len, device=device).unsqueeze(0)

        x    = (self.token_embedding(input_ids) +
                self.pos_embedding(pos_ids))
        x    = self.dropout(x)

        # Causal mask
        mask = self.create_causal_mask(seq_len, device)

        for layer in self.layers:
            x = layer(x, mask)

        x      = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # USE THIS
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=0

            )

        return logits, loss

# Test GPT model
gpt = GPTModel(
    vocab_size  = 1000,
    d_model     = 64,
    num_heads   = 4,
    d_ff        = 128,
    num_layers  = 2,
    max_len     = 128
)

total_params = sum(p.numel() for p in gpt.parameters())
print(f"GPT Style Model:")
print(f"Parameters       : {total_params:,}")

input_ids = torch.randint(1, 1000, (2, 10))
targets   = torch.randint(1, 1000, (2, 10))

logits, loss = gpt(input_ids, targets)
print(f"Input shape      : {input_ids.shape}")
print(f"Output shape     : {logits.shape}")
print(f"Training loss    : {loss.item():.4f}")


# ------------------------------------------
# PART 5 - FINE TUNING BERT
# ------------------------------------------

print("\n===== PART 5: Fine Tuning BERT =====")

print("""
FINE TUNING:
    Take pretrained BERT model
    Add task specific head on top
    Train on small labeled dataset
    Much better than training from scratch

BERT FINE TUNING FOR CLASSIFICATION:
    Input  : [CLS] sentence [SEP]
    Output : CLS token representation
    Head   : Linear layer -> num_classes

BERT FINE TUNING FOR NER:
    Input  : [CLS] token1 token2 ... [SEP]
    Output : representation for each token
    Head   : Linear -> num_entity_types per token

BERT FINE TUNING FOR QA:
    Input  : [CLS] question [SEP] context [SEP]
    Output : start and end position of answer span
    Head   : Linear -> 2 (start, end logits)

WHY FINE TUNING WORKS:
    BERT already knows language
    Fine tuning just adapts to specific task
    Need very little labeled data
    Fast training - few epochs enough
""")

# Fine tuning simulation
class BERTForSentimentAnalysis(nn.Module):
    def __init__(self, vocab_size, d_model=64,
                 num_heads=4, d_ff=128,
                 num_layers=2, dropout=0.1):
        super(BERTForSentimentAnalysis, self).__init__()

        self.bert    = BERTModel(
            vocab_size, d_model, num_heads,
            d_ff, num_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)
        )

    def forward(self, input_ids, attention_mask=None):
        _, cls_out = self.bert(
            input_ids,
            attention_mask=attention_mask
        )
        cls_out = self.dropout(cls_out)
        return self.classifier(cls_out)


# Dataset
positive_reviews = [
    "this is an absolutely fantastic product",
    "loved every single moment of using this",
    "excellent quality and amazing performance",
    "highly recommend this wonderful product",
    "outstanding results very happy customer",
    "perfect product exceeded all expectations",
    "brilliant design and great functionality",
    "amazing value for money highly satisfied",
]

negative_reviews = [
    "terrible product complete waste of money",
    "worst purchase ever do not buy this",
    "awful quality very disappointed customer",
    "horrible experience never buying again",
    "complete garbage does not work at all",
    "very poor quality extremely disappointing",
    "waste of money stopped working immediately",
    "awful product totally not recommended",
]

reviews = positive_reviews + negative_reviews
labels  = [1] * len(positive_reviews) + [0] * len(negative_reviews)

# Build vocabulary with special tokens
vocab   = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[MASK]": 3}
idx     = 4
for review in reviews:
    for word in review.split():
        if word not in vocab:
            vocab[word] = idx
            idx        += 1

def encode_for_bert(text, vocab, max_len=16):
    tokens  = ["[CLS]"] + text.split()[:max_len-2] + ["[SEP]"]
    ids     = [vocab.get(t, 0) for t in tokens]
    mask    = [1] * len(ids)
    # Pad
    pad_len = max_len - len(ids)
    ids    += [0] * pad_len
    mask   += [0] * pad_len
    return ids, mask

max_len  = 16
encodings = [encode_for_bert(r, vocab, max_len) for r in reviews]

X        = torch.LongTensor([e[0] for e in encodings])
masks    = torch.FloatTensor([e[1] for e in encodings])
y        = torch.LongTensor(labels)

# Split
split    = int(0.8 * len(reviews))
X_train  = X[:split]
m_train  = masks[:split]
y_train  = y[:split]
X_test   = X[split:]
m_test   = masks[split:]
y_test   = y[split:]

# Model
model    = BERTForSentimentAnalysis(
    vocab_size = len(vocab),
    d_model    = 32,
    num_heads  = 4,
    d_ff       = 64,
    num_layers = 2,
    dropout    = 0.1
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=2e-4,
    weight_decay=1e-4
)
criterion = nn.CrossEntropyLoss()

total_params = sum(p.numel() for p in model.parameters())
print(f"Vocabulary size  : {len(vocab)}")
print(f"Model Parameters : {total_params:,}")
print(f"\nFine-tuning BERT for Sentiment Analysis...")
print(f"{'Epoch':<8} {'Loss':<12} {'Train Acc':<12} {'Test Acc'}")
print("-" * 45)

for epoch in range(200):
    model.train()
    optimizer.zero_grad()

    logits = model(X_train, m_train)
    loss   = criterion(logits, y_train)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=1.0)
    optimizer.step()

    if epoch % 40 == 0:
        model.eval()
        with torch.no_grad():
            train_preds = model(X_train, m_train).argmax(dim=-1)
            test_preds  = model(X_test, m_test).argmax(dim=-1)
            train_acc   = (train_preds == y_train).float().mean()
            test_acc    = (test_preds == y_test).float().mean()

        print(f"{epoch:<8} {loss.item():<12.4f} "
              f"{train_acc.item():<12.4f} {test_acc.item():.4f}")

# Final predictions
model.eval()
with torch.no_grad():
    test_preds = model(X_test, m_test).argmax(dim=-1)
    final_acc  = (test_preds == y_test).float().mean()

print(f"\nFinal Test Accuracy : {final_acc.item()*100:.2f}%")


# ------------------------------------------
# PART 6 - GPT TEXT GENERATION
# ------------------------------------------

print("\n===== PART 6: GPT Text Generation =====")

print("""
TEXT GENERATION WITH GPT:
    Start with a prompt
    Predict next token probabilities
    Sample from distribution
    Append to input
    Repeat until done

TEMPERATURE:
    Controls randomness of generation
    Low  temperature (0.1) - conservative, repetitive
    High temperature (1.5) - creative, sometimes nonsense
    Temperature 1.0        - use model distribution as is

TOP-K SAMPLING:
    Only sample from top K most likely tokens
    Prevents very unlikely tokens
    k=50 is common default

TOP-P SAMPLING (Nucleus):
    Sample from smallest set of tokens
    whose cumulative probability exceeds p
    p=0.9 means sample from tokens that together
    account for 90% of probability mass
    More dynamic than top-k
""")

# Simple text generation with our GPT model
def generate_text(model, vocab, start_text,
                  max_new_tokens=20, temperature=1.0):
    model.eval()

    # Reverse vocab for decoding
    idx_to_word = {v: k for k, v in vocab.items()}

    # Encode start text
    tokens = [vocab.get(w, 1) for w in start_text.split()]
    tokens = torch.LongTensor(tokens).unsqueeze(0)

    generated = start_text.split()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(tokens)

            # Get last token logits
            next_logits = logits[0, -1, :] / temperature

            # Softmax to probabilities
            probs = F.softmax(next_logits, dim=-1)

            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)
            next_word  = idx_to_word.get(
                next_token.item(), "<UNK>")

            generated.append(next_word)
            tokens = torch.cat(
                [tokens, next_token.unsqueeze(0)], dim=-1)

            # Stop at padding
            if next_token.item() == 0:
                break

    return " ".join(generated)

# Train small GPT on simple dataset
training_texts = [
    "the cat sat on the mat",
    "the dog ran in the park",
    "machine learning is very powerful",
    "deep learning uses neural networks",
    "transformers changed the world of nlp",
    "bert uses encoder for understanding text",
    "gpt uses decoder for generating text",
    "attention mechanism is very important",
    "python is popular for machine learning",
    "prateek is learning machine learning well",
]

# Build vocab
gpt_vocab  = {"<PAD>": 0, "<UNK>": 1}
gpt_idx    = 2
for text in training_texts:
    for word in text.split():
        if word not in gpt_vocab:
            gpt_vocab[word] = gpt_idx
            gpt_idx        += 1

# Encode
def encode_gpt(text, vocab, max_len=10):
    tokens = [vocab.get(w, 1) for w in text.split()]
    tokens = tokens[:max_len]
    tokens+= [0] * (max_len - len(tokens))
    return tokens

max_len   = 10
sequences = torch.LongTensor(
    [encode_gpt(t, gpt_vocab, max_len) for t in training_texts])

# Input is sequence except last token
# Target is sequence shifted by one
inputs  = sequences[:, :-1]
targets = sequences[:, 1:]

# Small GPT model
small_gpt = GPTModel(
    vocab_size  = len(gpt_vocab),
    d_model     = 32,
    num_heads   = 2,
    d_ff        = 64,
    num_layers  = 2,
    max_len     = 64
)

optimizer = torch.optim.Adam(
    small_gpt.parameters(), lr=0.001)

print(f"Training small GPT language model...")
print(f"Vocabulary size  : {len(gpt_vocab)}")
print(f"Training texts   : {len(training_texts)}")
print(f"{'Epoch':<10} {'Loss'}")
print("-" * 25)

for epoch in range(500):
    small_gpt.train()
    optimizer.zero_grad()

    logits, loss = small_gpt(inputs, targets)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        small_gpt.parameters(), max_norm=1.0)
    optimizer.step()

    if epoch % 100 == 0:
        print(f"{epoch:<10} {loss.item():.4f}")

# Generate text
print(f"\nText Generation Examples:")
prompts = [
    "the cat",
    "machine learning",
    "transformers changed",
    "prateek is",
]

for prompt in prompts:
    generated = generate_text(
        small_gpt, gpt_vocab, prompt,
        max_new_tokens=5, temperature=0.8)
    print(f"  Prompt   : {prompt}")
    print(f"  Generated: {generated}")
    print()


# ------------------------------------------
# MINI PROJECT - Complete NLP Pipeline
# ------------------------------------------

print("===== MINI PROJECT: Complete NLP Pipeline =====")

print("""
Combining everything from Week 4:
    Text cleaning and tokenization
    Word embeddings (learned)
    BERT style encoder
    Multi class text classification
""")

# Multi class dataset
tech_texts = [
    "python machine learning deep learning neural networks",
    "tensorflow pytorch transformers bert gpt models",
    "convolutional networks image classification computer vision",
]
sports_texts = [
    "cricket football basketball tennis swimming sports",
    "world cup championship tournament league players team",
    "athlete training performance fitness competition medal",
]
food_texts = [
    "cooking recipe ingredients kitchen restaurant delicious",
    "pizza pasta salad burger sandwich healthy meal",
    "vegetarian vegan nutrition diet calories protein fiber",
]

all_texts  = tech_texts + sports_texts + food_texts
all_labels = [0]*3 + [1]*3 + [2]*3
class_names = ["Technology", "Sports", "Food"]

# Build vocab
pipeline_vocab = {"[PAD]": 0, "[CLS]": 1}
pidx           = 2
for text in all_texts:
    for word in text.split():
        if word not in pipeline_vocab:
            pipeline_vocab[word] = pidx
            pidx += 1

def encode_pipeline(text, vocab, max_len=12):
    tokens = ["[CLS]"] + text.split()[:max_len-1]
    ids    = [vocab.get(t, 0) for t in tokens]
    ids   += [0] * (max_len - len(ids))
    return ids

max_len  = 12
X_pipe   = torch.LongTensor(
    [encode_pipeline(t, pipeline_vocab, max_len)
     for t in all_texts])
y_pipe   = torch.LongTensor(all_labels)

# BERT for multi class
pipeline_model = BERTForClassification(
    vocab_size  = len(pipeline_vocab),
    d_model     = 32,
    num_heads   = 2,
    d_ff        = 64,
    num_layers  = 2,
    num_classes = 3
)

class BERTForClassification(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads,
                 d_ff, num_layers, num_classes, dropout=0.1):
        super(BERTForClassification, self).__init__()
        self.bert = BERTModel(
            vocab_size, d_model, num_heads,
            d_ff, num_layers, dropout=dropout)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids, attention_mask=None):
        _, cls_out = self.bert(
            input_ids, attention_mask=attention_mask)
        return self.classifier(self.dropout(cls_out))

pipeline_model  = BERTForClassification(
    vocab_size  = len(pipeline_vocab),
    d_model     = 32,
    num_heads   = 2,
    d_ff        = 64,
    num_layers  = 2,
    num_classes = 3
)

optimizer = torch.optim.Adam(
    pipeline_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print(f"Training multi-class text classifier...")
for epoch in range(300):
    pipeline_model.train()
    optimizer.zero_grad()
    logits = pipeline_model(X_pipe)
    loss   = criterion(logits, y_pipe)
    loss.backward()
    optimizer.step()

pipeline_model.eval()
with torch.no_grad():
    preds    = pipeline_model(X_pipe).argmax(dim=-1)
    accuracy = (preds == y_pipe).float().mean()

print(f"Final Accuracy   : {accuracy.item()*100:.2f}%")

# Test new texts
new_texts = [
    "neural network transformer attention mechanism",
    "football world cup championship tournament",
    "vegetarian recipe healthy cooking ingredients",
]

print(f"\nNew Text Predictions:")
print(f"{'Text':<45} {'Prediction'}")
print("-" * 60)
for text in new_texts:
    seq    = torch.LongTensor(
        [encode_pipeline(text, pipeline_vocab, max_len)])
    with torch.no_grad():
        logits = pipeline_model(seq)
        pred   = logits.argmax(dim=-1).item()
    print(f"{text[:43]:<45} {class_names[pred]}")


print("\n===== WHAT I LEARNED TODAY =====")
print("BERT - bidirectional encoder for understanding")
print("GPT  - causal decoder for generation")
print("MLM  - masked language modeling pretraining")
print("WordPiece and BPE tokenization")
print("BERT fine tuning for classification")
print("GPT text generation with temperature")
print("Top-k and top-p sampling strategies")
print("Complete NLP pipeline from scratch")
print("\nDay 27 Done! Tomorrow is REST DAY!")