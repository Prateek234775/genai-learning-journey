# ============================================
# DAY 26 - Transformer Architecture
# Encoder, Decoder, Self Attention
# Author: Prateek Kumar Kuntal
# Date: 30 May 2025
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# ------------------------------------------
# PART 1 - WHAT IS A TRANSFORMER
# ------------------------------------------

print("===== PART 1: What is a Transformer =====")

print("""
TRANSFORMER:
    Introduced in "Attention is All You Need" 2017
    by Vaswani et al. at Google Brain

    Replaced RNNs and LSTMs completely
    No recurrence - processes all tokens in parallel
    Uses attention mechanism exclusively

WHY TRANSFORMERS WON:
    RNN - sequential processing, slow training
    Transformer - parallel processing, fast training
    Can scale to billions of parameters
    Long range dependencies captured perfectly

ARCHITECTURE:
    Encoder Stack - understands input
    Decoder Stack - generates output

    Each encoder layer:
        Multi Head Self Attention
        Feed Forward Network
        Layer Norm and Residual connections

    Each decoder layer:
        Masked Multi Head Self Attention
        Cross Attention (attends to encoder output)
        Feed Forward Network
        Layer Norm and Residual connections

FAMOUS MODELS BUILT ON TRANSFORMER:
    BERT      - encoder only  - understanding tasks
    GPT       - decoder only  - generation tasks
    T5        - encoder decoder - translation, summarization
    ChatGPT   - decoder only  - conversation
    Claude    - decoder only  - conversation
    Gemini    - decoder only  - conversation
""")


# ------------------------------------------
# PART 2 - BUILDING BLOCKS RECAP
# ------------------------------------------

print("===== PART 2: Building Blocks =====")

# Multi Head Attention from yesterday
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0

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
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.num_heads, self.d_k)
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
        output  = self.W_o(output)

        return output, weights

# Feed Forward Network
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.network(x)

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(dropout)

        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() *
            -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe          = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

print("Building blocks ready:")
print("  Multi Head Attention")
print("  Feed Forward Network")
print("  Positional Encoding")


# ------------------------------------------
# PART 3 - ENCODER LAYER
# ------------------------------------------

print("\n===== PART 3: Encoder Layer =====")

print("""
ENCODER LAYER:
    Two sub layers:
    1. Multi Head Self Attention
    2. Feed Forward Network

    Each sub layer wrapped with:
    - Residual connection (add input to output)
    - Layer normalization

    output = LayerNorm(x + SubLayer(x))

RESIDUAL CONNECTIONS:
    Add input directly to output
    Prevents vanishing gradient
    Allows gradients to flow freely
    Used in ResNet too (remember Day 20!)

LAYER NORMALIZATION:
    Normalize across features not batch
    More stable for sequence models than BatchNorm
    Applied after residual connection
""")

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(
            d_model, num_heads, dropout)
        self.feed_forward   = FeedForward(
            d_model, d_ff, dropout)

        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # Sub layer 1 - self attention with residual
        attn_out, _ = self.self_attention(x, x, x, src_mask)
        x           = self.norm1(x + self.dropout(attn_out))

        # Sub layer 2 - feed forward with residual
        ff_out      = self.feed_forward(x)
        x           = self.norm2(x + ff_out)

        return x

# Test encoder layer
batch    = 2
seq_len  = 10
d_model  = 32
num_heads= 4
d_ff     = 64

enc_layer = EncoderLayer(d_model, num_heads, d_ff)
x         = torch.randn(batch, seq_len, d_model)
enc_out   = enc_layer(x)

print(f"Encoder Layer:")
print(f"Input shape      : {x.shape}")
print(f"Output shape     : {enc_out.shape}")
print(f"Shape preserved  : {x.shape == enc_out.shape}")
print(f"Parameters       : "
      f"{sum(p.numel() for p in enc_layer.parameters()):,}")


# ------------------------------------------
# PART 4 - DECODER LAYER
# ------------------------------------------

print("\n===== PART 4: Decoder Layer =====")

print("""
DECODER LAYER:
    Three sub layers:
    1. Masked Multi Head Self Attention
       Causal mask prevents attending to future tokens
       Used during training to predict next token

    2. Cross Attention
       Query from decoder
       Key and Value from encoder output
       Decoder attends to relevant encoder positions
       This is how translation works!

    3. Feed Forward Network

    Each wrapped with residual and layer norm

CROSS ATTENTION INTUITION:
    Translating "I love learning" to Hindi
    When generating each Hindi word
    Decoder queries which English words are relevant
    Encoder keys and values provide English context
    Attention weights show alignment between languages
""")

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()

        # Masked self attention
        self.self_attention  = MultiHeadAttention(
            d_model, num_heads, dropout)

        # Cross attention
        self.cross_attention = MultiHeadAttention(
            d_model, num_heads, dropout)

        # Feed forward
        self.feed_forward    = FeedForward(
            d_model, d_ff, dropout)

        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.norm3   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output,
                src_mask=None, tgt_mask=None):
        # Sub layer 1 - masked self attention
        attn_out, _ = self.self_attention(
            x, x, x, tgt_mask)
        x           = self.norm1(x + self.dropout(attn_out))

        # Sub layer 2 - cross attention
        cross_out, cross_weights = self.cross_attention(
            x, enc_output, enc_output, src_mask)
        x           = self.norm2(x + self.dropout(cross_out))

        # Sub layer 3 - feed forward
        ff_out      = self.feed_forward(x)
        x           = self.norm3(x + ff_out)

        return x, cross_weights

# Test decoder layer
dec_layer  = DecoderLayer(d_model, num_heads, d_ff)
tgt        = torch.randn(batch, 8, d_model)
dec_out, cross_w = dec_layer(tgt, enc_out)

print(f"Decoder Layer:")
print(f"Target input shape   : {tgt.shape}")
print(f"Encoder output shape : {enc_out.shape}")
print(f"Decoder output shape : {dec_out.shape}")
print(f"Cross attention shape: {cross_w.shape}")
print(f"Parameters           : "
      f"{sum(p.numel() for p in dec_layer.parameters()):,}")


# ------------------------------------------
# PART 5 - FULL TRANSFORMER
# ------------------------------------------

print("\n===== PART 5: Full Transformer =====")

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads,
                 d_ff, num_layers, max_len=512, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.embedding  = nn.Embedding(vocab_size, d_model,
                                        padding_idx=0)
        self.pos_enc    = PositionalEncoding(
            d_model, max_len, dropout)
        self.layers     = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm       = nn.LayerNorm(d_model)
        self.scale      = math.sqrt(d_model)

    def forward(self, src, src_mask=None):
        x = self.pos_enc(self.embedding(src) * self.scale)

        for layer in self.layers:
            x = layer(x, src_mask)

        return self.norm(x)


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads,
                 d_ff, num_layers, max_len=512, dropout=0.1):
        super(TransformerDecoder, self).__init__()

        self.embedding  = nn.Embedding(vocab_size, d_model,
                                        padding_idx=0)
        self.pos_enc    = PositionalEncoding(
            d_model, max_len, dropout)
        self.layers     = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm       = nn.LayerNorm(d_model)
        self.scale      = math.sqrt(d_model)

    def forward(self, tgt, enc_output,
                src_mask=None, tgt_mask=None):
        x = self.pos_enc(self.embedding(tgt) * self.scale)

        for layer in self.layers:
            x, _ = layer(x, enc_output, src_mask, tgt_mask)

        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model,
                 num_heads, d_ff, num_layers,
                 max_len=512, dropout=0.1):
        super(Transformer, self).__init__()

        self.encoder    = TransformerEncoder(
            src_vocab, d_model, num_heads,
            d_ff, num_layers, max_len, dropout)

        self.decoder    = TransformerDecoder(
            tgt_vocab, d_model, num_heads,
            d_ff, num_layers, max_len, dropout)

        self.output_proj = nn.Linear(d_model, tgt_vocab)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def create_causal_mask(self, seq_len, device):
        mask = torch.tril(
            torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, src, tgt,
                src_mask=None, tgt_mask=None):
        enc_output = self.encoder(src, src_mask)

        if tgt_mask is None:
            tgt_mask = self.create_causal_mask(
                tgt.size(1), tgt.device)

        dec_output = self.decoder(
            tgt, enc_output, src_mask, tgt_mask)

        logits     = self.output_proj(dec_output)

        return logits

# Build transformer
src_vocab  = 1000
tgt_vocab  = 1000
d_model    = 64
num_heads  = 4
d_ff       = 128
num_layers = 2

transformer = Transformer(
    src_vocab, tgt_vocab, d_model,
    num_heads, d_ff, num_layers
)

total_params = sum(p.numel() for p in transformer.parameters())
print(f"Full Transformer Architecture:")
print(f"Source vocab     : {src_vocab}")
print(f"Target vocab     : {tgt_vocab}")
print(f"d_model          : {d_model}")
print(f"Num heads        : {num_heads}")
print(f"d_ff             : {d_ff}")
print(f"Num layers       : {num_layers}")
print(f"Total Parameters : {total_params:,}")

# Test forward pass
src    = torch.randint(1, src_vocab, (2, 10))
tgt    = torch.randint(1, tgt_vocab, (2, 8))
logits = transformer(src, tgt)

print(f"\nForward Pass:")
print(f"Source shape     : {src.shape}")
print(f"Target shape     : {tgt.shape}")
print(f"Output shape     : {logits.shape}")
print(f"(batch=2, tgt_len=8, vocab=1000)")


# ------------------------------------------
# PART 6 - ENCODER ONLY vs DECODER ONLY
# ------------------------------------------

print("\n===== PART 6: Encoder Only vs Decoder Only =====")

print("""
ENCODER ONLY (BERT style):
    Reads entire input at once
    Bidirectional - sees past and future
    Good for understanding tasks:
        Text classification
        Named entity recognition
        Question answering (extractive)
        Sentence similarity

DECODER ONLY (GPT style):
    Generates one token at a time
    Causal - only sees past tokens
    Good for generation tasks:
        Text generation
        Conversation
        Code generation
        Story writing

ENCODER DECODER (T5 style):
    Encoder understands input
    Decoder generates output
    Good for transformation tasks:
        Translation
        Summarization
        Question answering (abstractive)

TODAY DECODER ONLY IS DOMINANT:
    GPT-3, GPT-4, Claude, Gemini, Llama
    All use decoder only architecture
    Simpler and scales better
    Can do understanding tasks too with prompting
""")

# Encoder only model (BERT style)
class BERTStyleEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads,
                 d_ff, num_layers, num_classes, dropout=0.1):
        super(BERTStyleEncoder, self).__init__()

        self.encoder = TransformerEncoder(
            vocab_size, d_model, num_heads,
            d_ff, num_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, mask=None):
        enc_out = self.encoder(x, mask)
        # Use CLS token (first token) for classification
        cls_out = enc_out[:, 0, :]
        cls_out = self.dropout(cls_out)
        return self.classifier(cls_out)

# Decoder only model (GPT style)
class GPTStyleDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads,
                 d_ff, num_layers, dropout=0.1):
        super(GPTStyleDecoder, self).__init__()

        self.embedding  = nn.Embedding(vocab_size, d_model,
                                        padding_idx=0)
        self.pos_enc    = PositionalEncoding(
            d_model, dropout=dropout)
        self.layers     = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm       = nn.LayerNorm(d_model)
        self.lm_head    = nn.Linear(d_model, vocab_size)
        self.scale      = math.sqrt(d_model)

    def forward(self, x):
        seq_len  = x.size(1)

        # Causal mask
        mask = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device)
        ).unsqueeze(0).unsqueeze(0)

        h = self.pos_enc(self.embedding(x) * self.scale)

        for layer in self.layers:
            h = layer(h, mask)

        h      = self.norm(h)
        logits = self.lm_head(h)

        return logits

# Test both
bert_model = BERTStyleEncoder(
    vocab_size=500, d_model=32, num_heads=4,
    d_ff=64, num_layers=2, num_classes=2)

gpt_model  = GPTStyleDecoder(
    vocab_size=500, d_model=32, num_heads=4,
    d_ff=64, num_layers=2)

src  = torch.randint(1, 500, (2, 10))
bert_out = bert_model(src)
gpt_out  = gpt_model(src)

print(f"BERT style encoder:")
print(f"  Input shape    : {src.shape}")
print(f"  Output shape   : {bert_out.shape}")
print(f"  (2 classes for classification)")
print(f"  Parameters     : "
      f"{sum(p.numel() for p in bert_model.parameters()):,}")

print(f"\nGPT style decoder:")
print(f"  Input shape    : {src.shape}")
print(f"  Output shape   : {gpt_out.shape}")
print(f"  (500 vocab for next token prediction)")
print(f"  Parameters     : "
      f"{sum(p.numel() for p in gpt_model.parameters()):,}")


# ------------------------------------------
# MINI PROJECT - Text Classifier with Transformer
# ------------------------------------------

print("\n===== MINI PROJECT: Text Classifier with Transformer =====")

# Dataset
positive = [
    "this product is absolutely amazing and works perfectly",
    "excellent quality highly recommend to everyone",
    "best purchase i have made love it so much",
    "fantastic product great value for money",
    "outstanding performance exceeded all expectations",
    "wonderful experience very happy with this product",
    "superb quality and fast delivery highly satisfied",
    "great product does exactly what it promises",
    "love this product works better than expected",
    "incredible quality amazing customer service too",
]

negative = [
    "terrible product complete waste of money",
    "worst purchase ever do not buy this garbage",
    "awful quality broke after one day of use",
    "very disappointed with this poor product",
    "horrible experience never buying from here again",
    "complete garbage does not work as advertised",
    "terrible quality extremely disappointing purchase",
    "waste of money product stopped working immediately",
    "awful product very poor quality not recommended",
    "horrible waste of time and money avoid this",
]

texts  = positive + negative
labels = [1] * len(positive) + [0] * len(negative)

# Vocabulary
vocab  = {"<PAD>": 0, "<CLS>": 1}
idx    = 2
for text in texts:
    for word in text.split():
        if word not in vocab:
            vocab[word] = idx
            idx        += 1

def encode(text, vocab, max_len=12):
    tokens = ["<CLS>"] + text.split()[:max_len-1]
    seq    = [vocab.get(t, 0) for t in tokens]
    seq   += [0] * (max_len - len(seq))
    return seq

max_len  = 12
X        = torch.LongTensor([encode(t, vocab, max_len)
                              for t in texts])
y        = torch.LongTensor(labels)

# Split
split    = int(0.8 * len(texts))
X_train  = X[:split]
y_train  = y[:split]
X_test   = X[split:]
y_test   = y[split:]

# Model
model    = BERTStyleEncoder(
    vocab_size  = len(vocab),
    d_model     = 32,
    num_heads   = 4,
    d_ff        = 64,
    num_layers  = 2,
    num_classes = 2,
    dropout     = 0.1
)

optimizer = torch.optim.Adam(
    model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=50, gamma=0.5)

total_params = sum(p.numel() for p in model.parameters())
print(f"Vocabulary size  : {len(vocab)}")
print(f"Model Parameters : {total_params:,}")
print(f"\nTraining Transformer Classifier...")
print(f"{'Epoch':<8} {'Loss':<12} {'Train Acc':<12} {'Test Acc'}")
print("-" * 45)

epochs = 200
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    logits = model(X_train)
    loss   = criterion(logits, y_train)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    if epoch % 40 == 0:
        model.eval()
        with torch.no_grad():
            train_logits = model(X_train)
            test_logits  = model(X_test)
            train_acc    = (train_logits.argmax(dim=-1) ==
                            y_train).float().mean()
            test_acc     = (test_logits.argmax(dim=-1) ==
                            y_test).float().mean()

        print(f"{epoch:<8} {loss.item():<12.4f} "
              f"{train_acc.item():<12.4f} {test_acc.item():.4f}")

# Final predictions
model.eval()
with torch.no_grad():
    test_logits = model(X_test)
    test_preds  = test_logits.argmax(dim=-1)
    final_acc   = (test_preds == y_test).float().mean()

print(f"\nFinal Test Accuracy : {final_acc.item()*100:.2f}%")

# Test new reviews
new_reviews = [
    "absolutely amazing product highly recommend everyone",
    "terrible quality complete waste do not buy",
    "great value for money very satisfied customer",
    "horrible product broke immediately very disappointed",
]

print(f"\nNew Review Predictions:")
print(f"{'Review':<50} {'Prediction':<12} {'Confidence'}")
print("-" * 75)

label_map = {0: "Negative", 1: "Positive"}
for review in new_reviews:
    seq    = torch.LongTensor([encode(review, vocab, max_len)])
    with torch.no_grad():
        logits = model(seq)
        probs  = F.softmax(logits, dim=-1)
        pred   = probs.argmax(dim=-1).item()
        conf   = probs.max().item()

    print(f"{review[:48]:<50} {label_map[pred]:<12} {conf*100:.1f}%")


print("\n===== WHAT I LEARNED TODAY =====")
print("Transformer architecture - encoder decoder")
print("Encoder layer - self attention and feed forward")
print("Decoder layer - masked self attention and cross attention")
print("Residual connections - prevent vanishing gradient")
print("Layer normalization - stable training")
print("Encoder only - BERT style for understanding")
print("Decoder only - GPT style for generation")
print("Full transformer for sequence to sequence")
print("Mini Project - Text Classifier with Transformer")
print("\nDay 26 Done! Tomorrow - BERT and GPT!")