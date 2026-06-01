# ============================================
# DAY 25 - Attention Mechanism
# The Key Idea Behind Transformers
# Author: Prateek Kumar Kuntal
# Date: 29 May 2025
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ------------------------------------------
# PART 1 - PROBLEM WITH RNN AND LSTM
# ------------------------------------------

print("===== PART 1: Problem with RNN and LSTM =====")

print("""
ENCODER DECODER ARCHITECTURE:
    Used for sequence to sequence tasks
    Translation, summarization, question answering

    Encoder reads input sentence
    Compresses entire sentence into one vector
    Decoder generates output from that vector

THE BOTTLENECK PROBLEM:
    Entire input sentence compressed into single vector
    For long sentences this vector loses information
    "The cat sat on the mat and it was happy"
    All this information in one small vector?

    Decoder has no direct access to input words
    Only sees the compressed summary vector
    Performance drops badly for long sentences

ATTENTION SOLVES THIS:
    Decoder can look at ALL encoder hidden states
    Not just the final compressed vector
    Learns which input words are relevant for each output word
    Like a human translator looking back at source text

REAL EXAMPLE:
    Translating "I love machine learning" to Hindi
    When generating "machine" the model should
    attend strongly to the word "machine" in input
    When generating "love" attend to "love" in input
    Attention scores tell the model where to look
""")


# ------------------------------------------
# PART 2 - ATTENTION FROM SCRATCH
# ------------------------------------------

print("===== PART 2: Attention from Scratch =====")

print("""
ATTENTION MECHANISM:
    Given a query and a set of key-value pairs
    Compute weighted sum of values
    Weights determined by similarity of query to keys

INTUITION:
    Query   = what I am looking for
    Keys    = what each item is about
    Values  = actual content of each item

    Like a search engine:
    Query   = your search term
    Keys    = webpage titles
    Values  = webpage content
    Attention = relevance weighted combination

STEPS:
    1. Compute similarity scores (query dot keys)
    2. Scale scores by sqrt(d_k) for stability
    3. Apply softmax to get probabilities
    4. Weighted sum of values using probabilities

    Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
""")

def attention(query, key, value, mask=None):
    d_k    = query.shape[-1]

    # Step 1 - compute similarity scores
    # USE THIS
    scores = query @ key.T

    # Step 2 - scale scores
    scores = scores / np.sqrt(d_k)

    # Step 3 - apply mask if provided
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    # Step 4 - softmax to get attention weights
    scores_exp    = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attention_weights = scores_exp / scores_exp.sum(axis=-1, keepdims=True)

    # Step 5 - weighted sum of values
    output = attention_weights @ value

    return output, attention_weights

# Simple example
np.random.seed(42)

seq_len = 5
d_k     = 8
d_v     = 8

Q = np.random.randn(seq_len, d_k)
K = np.random.randn(seq_len, d_k)
V = np.random.randn(seq_len, d_v)

output, weights = attention(Q, K, V)

print(f"Query  shape     : {Q.shape}")
print(f"Key    shape     : {K.shape}")
print(f"Value  shape     : {V.shape}")
print(f"Output shape     : {output.shape}")
print(f"Weights shape    : {weights.shape}")

print(f"\nAttention Weights (each row sums to 1):")
print(weights.round(4))
print(f"\nRow sums (should all be 1.0):")
print(weights.sum(axis=-1).round(4))

print(f"\nOutput (weighted combination of values):")
print(output.round(4))


# ------------------------------------------
# PART 3 - SELF ATTENTION
# ------------------------------------------

print("\n===== PART 3: Self Attention =====")

print("""
SELF ATTENTION:
    Query, Key, Value all come from same sequence
    Each position attends to all other positions
    Captures relationships between words in same sentence

    "The animal did not cross because it was tired"
    What does "it" refer to? "animal" or "street"?
    Self attention helps resolve this ambiguity

    Each word learns:
    Which other words are relevant to understanding it
    "tired" strongly attends to "animal" not "street"

HOW IT WORKS:
    Input sequence X of shape (seq_len, d_model)
    Three weight matrices: W_Q, W_K, W_V
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    Attention(Q, K, V) computed as before

    W_Q, W_K, W_V are learned during training
    Model learns what questions to ask (Q)
    What to match against (K)
    What information to extract (V)
""")

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super(SelfAttention, self).__init__()

        self.d_model = d_model

        # Linear projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        self.scale = d_model ** 0.5

    def forward(self, x, mask=None):
        # x shape: (batch, seq_len, d_model)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Attention scores
        scores = (Q @ K.transpose(-2, -1)) / self.scale

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Attention weights
        weights = F.softmax(scores, dim=-1)

        # Output
        output  = weights @ V

        return output, weights

# Test self attention
batch_size = 2
seq_len    = 6
d_model    = 16

x            = torch.randn(batch_size, seq_len, d_model)
self_attn    = SelfAttention(d_model)
output, weights = self_attn(x)

print(f"Input shape      : {x.shape}")
print(f"Output shape     : {output.shape}")
print(f"Weights shape    : {weights.shape}")

print(f"\nAttention weights for first sample first head:")
print(weights[0].detach().numpy().round(3))
print(f"\nEach row sums to: {weights[0].sum(dim=-1).detach().numpy().round(4)}")


# ------------------------------------------
# PART 4 - MULTI HEAD ATTENTION
# ------------------------------------------

print("\n===== PART 4: Multi Head Attention =====")

print("""
MULTI HEAD ATTENTION:
    Run attention multiple times in parallel
    Each head learns different types of relationships

    Head 1 might focus on syntactic relationships
    Head 2 might focus on semantic relationships
    Head 3 might focus on positional relationships

    Results concatenated and projected

WHY MULTIPLE HEADS?
    Single attention captures one type of relationship
    Multiple heads capture diverse relationships
    More expressive and powerful

EXAMPLE with 8 heads:
    d_model = 512
    num_heads = 8
    d_k = d_model / num_heads = 64 per head

    Each head works in 64 dimensional space
    8 heads concatenated = 512 dimensional output
    Final linear projection = 512 dimensional output

FORMULA:
    MultiHead(Q,K,V) = Concat(head_1,...,head_h) @ W_O
    where head_i = Attention(Q @ W_Qi, K @ W_Ki, V @ W_Vi)
""")

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads

        # Linear projections for all heads at once
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.scale = self.d_k ** 0.5

    def split_heads(self, x):
        # x: (batch, seq_len, d_model)
        batch, seq_len, _ = x.shape
        # Reshape to (batch, num_heads, seq_len, d_k)
        x = x.view(batch, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(self, x, mask=None):
        batch = x.shape[0]

        # Linear projections
        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))

        # Scaled dot product attention
        scores  = (Q @ K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        weights = F.softmax(scores, dim=-1)
        output  = weights @ V

        # Concatenate heads
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch, -1, self.d_model)

        # Final projection
        output = self.W_o(output)

        return output, weights

# Test multi head attention
d_model   = 32
num_heads = 4
x         = torch.randn(2, 6, d_model)

mha          = MultiHeadAttention(d_model, num_heads)
output, weights = mha(x)

total_params = sum(p.numel() for p in mha.parameters())
print(f"d_model          : {d_model}")
print(f"num_heads        : {num_heads}")
print(f"d_k per head     : {d_model // num_heads}")
print(f"Input shape      : {x.shape}")
print(f"Output shape     : {output.shape}")
print(f"Weights shape    : {weights.shape}")
print(f"Parameters       : {total_params:,}")


# ------------------------------------------
# PART 5 - POSITIONAL ENCODING
# ------------------------------------------

print("\n===== PART 5: Positional Encoding =====")

print("""
PROBLEM WITH ATTENTION:
    Attention has no notion of word order
    "dog bites man" and "man bites dog"
    Would produce same attention if words same

    RNN processes sequentially so order implicit
    Attention processes all at once needs explicit position

POSITIONAL ENCODING:
    Add position information to word embeddings
    Each position gets unique encoding vector
    Added to word embedding before attention

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Sine and cosine at different frequencies
    Each position has unique pattern
    Model can learn relative positions
""")

def positional_encoding(seq_len, d_model):
    pe  = np.zeros((seq_len, d_model))
    pos = np.arange(seq_len).reshape(-1, 1)
    div = np.exp(
        np.arange(0, d_model, 2) *
        -(np.log(10000.0) / d_model)
    )

    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div)

    return pe

seq_len = 10
d_model = 16
pe      = positional_encoding(seq_len, d_model)

print(f"Positional Encoding shape: {pe.shape}")
print(f"\nFirst 3 positions (first 8 dims):")
print(pe[:3, :8].round(4))
print(f"\nEach position has unique encoding!")

# PyTorch positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(dropout)

        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() *
            -(np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe          = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

pos_enc = PositionalEncoding(d_model=32, max_len=100)
x       = torch.randn(2, 10, 32)
x_pe    = pos_enc(x)

print(f"\nInput shape      : {x.shape}")
print(f"After PE shape   : {x_pe.shape}")
print(f"Shape unchanged - only values modified by position")


# ------------------------------------------
# PART 6 - MASKED ATTENTION
# ------------------------------------------

print("\n===== PART 6: Masked Attention =====")

print("""
TWO TYPES OF MASKS:

1. PADDING MASK:
    Sequences in a batch have different lengths
    Shorter sequences padded with zeros
    We should not attend to padding tokens
    Padding mask = 0 for padding positions

2. CAUSAL MASK (Look-ahead mask):
    Used in decoder during training
    When predicting word at position t
    Should only see words before position t
    Cannot look into the future
    Upper triangular matrix of -infinity

    "I love learning"
    Predicting "love"    - can only see "I"
    Predicting "learning"- can only see "I love"
    This is how GPT is trained
""")

# Causal mask
def create_causal_mask(seq_len):
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask

seq_len    = 5
causal_mask = create_causal_mask(seq_len)
print("Causal Mask (1=can attend, 0=cannot attend):")
print(causal_mask.numpy().astype(int))
print("\nRow 0 (first word) can only see itself")
print("Row 4 (last word) can see all previous words")

# Padding mask
def create_padding_mask(sequences, pad_idx=0):
    return (sequences != pad_idx).unsqueeze(1).unsqueeze(2)

sequences  = torch.tensor([
    [5, 3, 2, 1, 0],   # last token is padding
    [4, 2, 8, 6, 9],   # no padding
    [7, 1, 0, 0, 0],   # last 3 tokens padding
])
pad_mask   = create_padding_mask(sequences)

print(f"\nPadding Mask shape: {pad_mask.shape}")
print("Sequence 1:")
print(sequences[0].numpy(), "-> mask:", pad_mask[0, 0, 0].numpy().astype(int))
print("Sequence 3:")
print(sequences[2].numpy(), "-> mask:", pad_mask[2, 0, 0].numpy().astype(int))


# ------------------------------------------
# MINI PROJECT - Attention Visualization
# Sentence Similarity with Attention
# ------------------------------------------

print("\n===== MINI PROJECT: Sentence Attention Analysis =====")

class AttentionAnalyzer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads):
        super(AttentionAnalyzer, self).__init__()

        self.embedding  = nn.Embedding(vocab_size, d_model,
                                        padding_idx=0)
        self.pos_enc    = PositionalEncoding(d_model)
        self.attention  = MultiHeadAttention(d_model, num_heads)
        self.norm       = nn.LayerNorm(d_model)
        self.fc         = nn.Linear(d_model, 2)

    def forward(self, x):
        emb     = self.pos_enc(self.embedding(x))
        out, w  = self.attention(emb)
        out     = self.norm(out + emb)
        pooled  = out.mean(dim=1)
        logits  = self.fc(pooled)
        return logits, w

# Build vocab and dataset
sentences = [
    "the movie was amazing and wonderful",
    "i loved the film very much",
    "terrible movie complete waste of time",
    "awful film i hated every moment",
    "great acting and wonderful story",
    "horrible acting and boring story",
]
sent_labels = [1, 1, 0, 0, 1, 0]

# Build vocab
vocab = {"<PAD>": 0}
idx   = 1
for sent in sentences:
    for word in sent.split():
        if word not in vocab:
            vocab[word] = idx
            idx        += 1

def encode(sent, vocab, max_len=8):
    tokens = sent.split()[:max_len]
    seq    = [vocab.get(t, 0) for t in tokens]
    seq   += [0] * (max_len - len(seq))
    return seq

max_len   = 8
X         = torch.LongTensor([encode(s, vocab, max_len)
                               for s in sentences])
y         = torch.LongTensor(sent_labels)

model     = AttentionAnalyzer(
    vocab_size = len(vocab),
    d_model    = 32,
    num_heads  = 4
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print(f"Vocabulary size  : {len(vocab)}")
print(f"Training Attention Analyzer...")
print(f"{'Epoch':<8} {'Loss':<12} {'Accuracy'}")
print("-" * 32)

for epoch in range(200):
    model.train()
    optimizer.zero_grad()

    logits, weights = model(X)
    loss            = criterion(logits, y)
    loss.backward()
    optimizer.step()

    if epoch % 40 == 0:
        preds    = logits.argmax(dim=-1)
        accuracy = (preds == y).float().mean()
        print(f"{epoch:<8} {loss.item():<12.4f} {accuracy.item():.4f}")

# Show attention weights
model.eval()
with torch.no_grad():
    logits, weights = model(X)
    preds           = logits.argmax(dim=-1)

print(f"\nAttention Analysis:")
print(f"{'Sentence':<45} {'True':<8} {'Pred'}")
print("-" * 60)

label_map = {0: "Negative", 1: "Positive"}
for i, sent in enumerate(sentences):
    true_label = label_map[sent_labels[i]]
    pred_label = label_map[preds[i].item()]
    correct    = "Correct" if sent_labels[i] == preds[i].item() else "Wrong"
    print(f"{sent[:43]:<45} {true_label:<8} {pred_label} {correct}")

# Show which words each position attends to
print(f"\nAttention Pattern for Sentence 1:")
print(f"'{sentences[0]}'")
words    = sentences[0].split()
attn_map = weights[0, 0].detach().numpy()

print(f"\n{'':>10}", end="")
for w in words:
    print(f"{w[:6]:>8}", end="")
print()

for i, word in enumerate(words):
    print(f"{word[:8]:>10}", end="")
    for j in range(len(words)):
        print(f"{attn_map[i,j]:>8.3f}", end="")
    print()


print("\n===== WHAT I LEARNED TODAY =====")
print("Bottleneck problem in encoder decoder")
print("Attention - query key value mechanism")
print("Self Attention - words attending to each other")
print("Multi Head Attention - multiple parallel heads")
print("Positional Encoding - inject position information")
print("Causal Mask - prevent looking into future")
print("Padding Mask - ignore padding tokens")
print("Mini Project - Attention Visualization")
print("\nDay 25 Done! Tomorrow - Transformer Architecture!")