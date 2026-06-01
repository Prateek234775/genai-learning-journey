# ============================================
# DAY 24 - RNNs and LSTMs
# How Sequence Models Work
# Author: Prateek Kumar Kuntal
# Date: 28 May 2025
# ============================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# ------------------------------------------
# PART 1 - WHAT IS A SEQUENCE MODEL
# ------------------------------------------

print("===== PART 1: What is a Sequence Model =====")

print("""
SEQUENCE MODELS:
    Models that process data in order
    Order matters unlike tabular data

    Examples of sequences:
        Text        - words in a sentence
        Time series - stock prices over time
        Audio       - sound waves over time
        Video       - frames over time
        DNA         - nucleotide sequences

WHY REGULAR NEURAL NETWORKS FAIL:
    Fixed input size - cannot handle variable length
    No memory - forgets previous inputs
    "I am happy" vs "I am not happy"
    NN treats these the same without word order

SEQUENCE MODELS REMEMBER:
    Process one element at a time
    Maintain hidden state (memory)
    Each step uses current input and past memory
    Can handle variable length sequences
""")


# ------------------------------------------
# PART 2 - RECURRENT NEURAL NETWORK
# ------------------------------------------

print("===== PART 2: Recurrent Neural Network =====")

print("""
RNN:
    Has a hidden state that acts as memory
    At each step receives current input and previous hidden state
    Produces output and new hidden state

    h_t = tanh(W_h * h_(t-1) + W_x * x_t + b)
    y_t = W_y * h_t + b_y

    h_t    = current hidden state
    h_(t-1)= previous hidden state
    x_t    = current input
    W_h    = weight for hidden state
    W_x    = weight for input

UNROLLING RNN:
    "I love learning"
    Step 1: input="I"      + h_0=zeros -> h_1
    Step 2: input="love"   + h_1       -> h_2
    Step 3: input="learning"+ h_2      -> h_3
    h_3 contains information about all 3 words

PROBLEM WITH VANILLA RNN:
    Vanishing Gradient - gradients shrink exponentially
    Cannot remember long term dependencies
    "The cat that sat on the mat was fat"
    By the time we reach "was" the model forgot "cat"
    LSTM solves this problem
""")

# Manual RNN implementation
class ManualRNN:
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        np.random.seed(42)

        # Weights
        self.W_x = np.random.randn(hidden_size, input_size) * 0.1
        self.W_h = np.random.randn(hidden_size, hidden_size) * 0.1
        self.b   = np.zeros((hidden_size, 1))

    def step(self, x, h_prev):
        # h_t = tanh(W_x @ x + W_h @ h_prev + b)
        z   = self.W_x @ x + self.W_h @ h_prev + self.b
        h_t = np.tanh(z)
        return h_t

    def forward(self, sequence):
        h = np.zeros((self.hidden_size, 1))
        hidden_states = []

        for x in sequence:
            x = x.reshape(-1, 1)
            h = self.step(x, h)
            hidden_states.append(h.copy())

        return hidden_states, h

# Test manual RNN
input_size  = 4
hidden_size = 8
rnn         = ManualRNN(input_size, hidden_size)

# Sequence of 5 timesteps
np.random.seed(42)
sequence = [np.random.randn(input_size) for _ in range(5)]

hidden_states, final_h = rnn.forward(sequence)

print("Manual RNN Forward Pass:")
print(f"Input size       : {input_size}")
print(f"Hidden size      : {hidden_size}")
print(f"Sequence length  : {len(sequence)}")
print(f"\nHidden state at each step:")
for i, h in enumerate(hidden_states):
    print(f"  Step {i+1}: shape={h.shape} "
          f"mean={h.mean():.4f} std={h.std():.4f}")
print(f"\nFinal hidden state (first 5 values):")
print(final_h.flatten()[:5].round(4))


# ------------------------------------------
# PART 3 - VANISHING GRADIENT PROBLEM
# ------------------------------------------

print("\n===== PART 3: Vanishing Gradient Problem =====")

print("""
VANISHING GRADIENT:
    During backpropagation gradients are multiplied
    at each timestep going backwards

    If gradient < 1 it shrinks exponentially
    0.9 ^ 100 = 0.000026 (almost zero!)

    Gradient at early timesteps becomes negligible
    Model cannot learn long term dependencies

EXAMPLE:
    "The cat that the dog chased was black"
    To predict "was" after "cat" requires
    remembering "cat" from 5 words back
    Vanilla RNN fails at this

    Long sequences = vanishing gradient = no learning

EXPLODING GRADIENT:
    Opposite problem - gradients grow exponentially
    NaN values in training
    Fixed with gradient clipping (already learned!)

LSTM solves vanishing gradient with gates
""")

# Demonstrate vanishing gradient
print("Vanishing Gradient Demonstration:")
print(f"{'Steps Back':<15} {'Gradient':<20} {'Effect'}")
print("-" * 50)

gradient = 1.0
factor   = 0.9     # typical gradient magnitude

for steps in [1, 5, 10, 20, 50, 100]:
    grad_at_step = factor ** steps
    effect       = ("Strong" if grad_at_step > 0.5 else
                    "Weak"   if grad_at_step > 0.1 else
                    "Tiny"   if grad_at_step > 0.01 else
                    "Gone")
    print(f"{steps:<15} {grad_at_step:<20.8f} {effect}")


# ------------------------------------------
# PART 4 - LSTM
# ------------------------------------------

print("\n===== PART 4: LSTM =====")

print("""
LSTM (Long Short Term Memory):
    Introduced by Hochreiter and Schmidhuber in 1997
    Solves vanishing gradient with gating mechanism
    Has two states: hidden state and cell state

CELL STATE:
    Long term memory highway
    Information flows with minimal modification
    Like a conveyor belt through the network

THREE GATES:
    1. FORGET GATE
       Decides what to forget from cell state
       f_t = sigmoid(W_f * [h_(t-1), x_t] + b_f)
       Output 0 = completely forget
       Output 1 = completely remember

    2. INPUT GATE
       Decides what new info to store
       i_t = sigmoid(W_i * [h_(t-1), x_t] + b_i)
       g_t = tanh(W_g * [h_(t-1), x_t] + b_g)

    3. OUTPUT GATE
       Decides what to output as hidden state
       o_t = sigmoid(W_o * [h_(t-1), x_t] + b_o)
       h_t = o_t * tanh(c_t)

CELL STATE UPDATE:
    c_t = f_t * c_(t-1) + i_t * g_t
    Forget some old info, add some new info

WHY LSTM WORKS:
    Cell state can carry info across many timesteps
    Gradients flow through cell state unchanged
    Gates control what to remember and forget
""")

# Manual LSTM implementation
class ManualLSTM:
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        np.random.seed(42)
        scale = 0.1

        combined = input_size + hidden_size

        # Forget gate
        self.W_f = np.random.randn(hidden_size, combined) * scale
        self.b_f = np.ones((hidden_size, 1))    # bias to 1 = remember

        # Input gate
        self.W_i = np.random.randn(hidden_size, combined) * scale
        self.b_i = np.zeros((hidden_size, 1))

        # Cell gate
        self.W_g = np.random.randn(hidden_size, combined) * scale
        self.b_g = np.zeros((hidden_size, 1))

        # Output gate
        self.W_o = np.random.randn(hidden_size, combined) * scale
        self.b_o = np.zeros((hidden_size, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def step(self, x, h_prev, c_prev):
        x      = x.reshape(-1, 1)
        h_prev = h_prev.reshape(-1, 1)
        c_prev = c_prev.reshape(-1, 1)

        # Concatenate input and hidden state
        combined = np.vstack([h_prev, x])

        # Forget gate
        f_t = self.sigmoid(self.W_f @ combined + self.b_f)

        # Input gate
        i_t = self.sigmoid(self.W_i @ combined + self.b_i)
        g_t = np.tanh(self.W_g @ combined + self.b_g)

        # Output gate
        o_t = self.sigmoid(self.W_o @ combined + self.b_o)

        # Cell state update
        c_t = f_t * c_prev + i_t * g_t

        # Hidden state
        h_t = o_t * np.tanh(c_t)

        return h_t, c_t, f_t, i_t, o_t

    def forward(self, sequence):
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))
        outputs = []

        for x in sequence:
            h, c, f, i, o = self.step(x, h, c)
            outputs.append({
                "h": h.copy(),
                "c": c.copy(),
                "forget": f.mean(),
                "input" : i.mean(),
                "output": o.mean(),
            })

        return outputs

# Test manual LSTM
lstm = ManualLSTM(input_size=4, hidden_size=8)

print("Manual LSTM Forward Pass:")
print(f"{'Step':<8} {'Forget Gate':<15} {'Input Gate':<15} "
      f"{'Output Gate':<15} {'Cell Mean'}")
print("-" * 65)

outputs = lstm.forward(sequence)
for i, out in enumerate(outputs):
    print(f"{i+1:<8} {out['forget']:<15.4f} {out['input']:<15.4f} "
          f"{out['output']:<15.4f} {out['c'].mean():.4f}")


# ------------------------------------------
# PART 5 - RNN AND LSTM IN PYTORCH
# ------------------------------------------

print("\n===== PART 5: RNN and LSTM in PyTorch =====")

# PyTorch RNN
rnn_layer = nn.RNN(
    input_size=10,
    hidden_size=20,
    num_layers=2,
    batch_first=True,
    dropout=0.2
)

# PyTorch LSTM
lstm_layer = nn.LSTM(
    input_size=10,
    hidden_size=20,
    num_layers=2,
    batch_first=True,
    dropout=0.2
)

# PyTorch GRU (simpler alternative to LSTM)
gru_layer = nn.GRU(
    input_size=10,
    hidden_size=20,
    num_layers=2,
    batch_first=True,
    dropout=0.2
)

print("PyTorch Sequence Models:")
print(f"\nRNN  parameters : "
      f"{sum(p.numel() for p in rnn_layer.parameters()):,}")
print(f"LSTM parameters : "
      f"{sum(p.numel() for p in lstm_layer.parameters()):,}")
print(f"GRU  parameters : "
      f"{sum(p.numel() for p in gru_layer.parameters()):,}")

# Test forward pass
batch_size = 4
seq_len    = 10
input_size = 10

x = torch.randn(batch_size, seq_len, input_size)

# RNN forward
rnn_out, h_n = rnn_layer(x)
print(f"\nRNN Input shape  : {x.shape}")
print(f"RNN Output shape : {rnn_out.shape}")
print(f"RNN Hidden shape : {h_n.shape}")

# LSTM forward
lstm_out, (h_n, c_n) = lstm_layer(x)
print(f"\nLSTM Input shape : {x.shape}")
print(f"LSTM Output shape: {lstm_out.shape}")
print(f"LSTM Hidden shape: {h_n.shape}")
print(f"LSTM Cell shape  : {c_n.shape}")

# GRU forward
gru_out, h_n = gru_layer(x)
print(f"\nGRU Input shape  : {x.shape}")
print(f"GRU Output shape : {gru_out.shape}")
print(f"GRU Hidden shape : {h_n.shape}")

print("""
batch_first=True means:
    Input shape  (batch, sequence, features)
    Output shape (batch, sequence, hidden_size)
    Last hidden  (num_layers, batch, hidden_size)
""")


# ------------------------------------------
# PART 6 - GRU
# ------------------------------------------

print("===== PART 6: GRU =====")

print("""
GRU (Gated Recurrent Unit):
    Simplified version of LSTM
    Introduced by Cho et al. in 2014
    Only 2 gates instead of 3

    UPDATE GATE - combines forget and input gates
    RESET GATE  - controls how much past to forget

LSTM vs GRU:
    LSTM - more expressive, more parameters
    GRU  - simpler, fewer parameters, faster
    GRU  - often performs as well as LSTM
    GRU  - preferred for smaller datasets

WHEN TO USE:
    Long sequences with complex dependencies -> LSTM
    Faster training needed                  -> GRU
    Small dataset                           -> GRU
    State of art NLP tasks                  -> Transformers
                                               (we cover tomorrow)
""")


# ------------------------------------------
# PART 7 - TYPES OF RNN ARCHITECTURES
# ------------------------------------------

print("===== PART 7: Types of RNN Architectures =====")

print("""
ONE TO ONE:
    Regular neural network
    Single input, single output
    Image classification

ONE TO MANY:
    Single input, sequence output
    Image captioning
    Music generation

MANY TO ONE:
    Sequence input, single output
    Sentiment analysis
    Text classification

MANY TO MANY (same length):
    Sequence input, sequence output
    Named entity recognition
    Part of speech tagging

MANY TO MANY (different length):
    Encoder decoder architecture
    Machine translation
    Summarization
    Question answering

BIDIRECTIONAL RNN:
    Process sequence forward and backward
    Each hidden state sees past and future context
    Used in BERT style models
""")

# Bidirectional LSTM
bi_lstm = nn.LSTM(
    input_size=10,
    hidden_size=20,
    num_layers=1,
    batch_first=True,
    bidirectional=True
)

x           = torch.randn(4, 10, 10)
bi_out, _   = bi_lstm(x)

print(f"Bidirectional LSTM:")
print(f"Input shape      : {x.shape}")
print(f"Output shape     : {bi_out.shape}")
print(f"(hidden_size * 2 because forward + backward)")
print(f"Parameters       : "
      f"{sum(p.numel() for p in bi_lstm.parameters()):,}")


# ------------------------------------------
# MINI PROJECT - Sentiment Analysis with LSTM
# ------------------------------------------

print("\n===== MINI PROJECT: Sentiment Analysis with LSTM =====")

# Dataset
positive_reviews = [
    "this movie was absolutely amazing and wonderful",
    "i loved every moment of this fantastic film",
    "great performance by all the actors in this movie",
    "highly recommend this beautiful and touching story",
    "one of the best movies i have ever seen",
    "excellent direction and outstanding cinematography",
    "this film made me laugh and cry with joy",
    "incredible story with amazing character development",
    "perfect movie for the whole family to enjoy",
    "brilliant screenplay with wonderful performances",
    "this was a masterpiece of modern cinema truly",
    "loved the music and the beautiful visual effects",
]

negative_reviews = [
    "this movie was terrible and a complete waste of time",
    "worst film i have ever seen in my life",
    "boring and predictable with bad acting throughout",
    "i hated every single moment of this awful movie",
    "completely disappointed by this horrible production",
    "terrible script and very poor direction overall",
    "do not waste your money on this bad film",
    "awful storyline with no character development at all",
    "this movie was so bad i left the theater early",
    "worst acting i have ever seen very disappointing",
    "complete garbage from start to finish avoid this",
    "terrible special effects and a nonsensical plot",
]

texts  = positive_reviews + negative_reviews
labels = [1] * len(positive_reviews) + [0] * len(negative_reviews)

# Build vocabulary
def build_vocab(texts):
    vocab  = {"<PAD>": 0, "<UNK>": 1}
    idx    = 2
    for text in texts:
        for word in text.lower().split():
            if word not in vocab:
                vocab[word] = idx
                idx        += 1
    return vocab

def text_to_sequence(text, vocab, max_len=15):
    tokens = text.lower().split()[:max_len]
    seq    = [vocab.get(t, 1) for t in tokens]
    # Pad to max_len
    seq   += [0] * (max_len - len(seq))
    return seq

vocab    = build_vocab(texts)
max_len  = 15

print(f"Vocabulary size  : {len(vocab)}")
print(f"Max sequence len : {max_len}")

# Convert to tensors
X = torch.LongTensor(
    [text_to_sequence(t, vocab, max_len) for t in texts])
y = torch.FloatTensor(labels)

# Split
split   = int(0.8 * len(texts))
X_train = X[:split]
y_train = y[:split]
X_test  = X[split:]
y_test  = y[split:]

print(f"Train size       : {len(X_train)}")
print(f"Test size        : {len(X_test)}")

# LSTM Sentiment Model
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim,
                 hidden_size, num_layers):
        super(SentimentLSTM, self).__init__()

        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=0)
        self.lstm      = nn.LSTM(
            embed_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        self.dropout   = nn.Dropout(0.5)
        self.fc        = nn.Linear(hidden_size * 2, 1)
        self.sigmoid   = nn.Sigmoid()

    def forward(self, x):
        embedded       = self.dropout(self.embedding(x))
        lstm_out, _    = self.lstm(embedded)
        # Use last timestep output
        last_out       = lstm_out[:, -1, :]
        out            = self.dropout(last_out)
        out            = self.fc(out)
        return self.sigmoid(out)

model     = SentimentLSTM(
    vocab_size  = len(vocab),
    embed_dim   = 32,
    hidden_size = 64,
    num_layers  = 2
)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model Parameters : {total_params:,}")

print(f"\nTraining LSTM Sentiment Classifier...")
print(f"{'Epoch':<8} {'Loss':<12} {'Train Acc':<12} {'Test Acc'}")
print("-" * 45)

epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    y_pred = model(X_train).squeeze()
    loss   = criterion(y_pred, y_train)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=1.0)
    optimizer.step()

    if epoch % 20 == 0:
        model.eval()
        with torch.no_grad():
            train_pred = model(X_train).squeeze()
            test_pred  = model(X_test).squeeze()
            train_acc  = ((train_pred > 0.5) ==
                          y_train.bool()).float().mean()
            test_acc   = ((test_pred > 0.5) ==
                          y_test.bool()).float().mean()

        print(f"{epoch:<8} {loss.item():<12.4f} "
              f"{train_acc.item():<12.4f} {test_acc.item():.4f}")

# Test predictions
model.eval()
test_reviews = [
    "this movie was absolutely wonderful and amazing",
    "terrible film complete waste of time and money",
    "great acting and wonderful story highly recommend",
    "boring and awful movie do not watch this film",
    "one of the best experiences i have ever had",
]

print(f"\nPredictions on New Reviews:")
print(f"{'Review':<50} {'Sentiment':<12} {'Confidence'}")
print("-" * 75)

for review in test_reviews:
    seq  = torch.LongTensor(
        [text_to_sequence(review, vocab, max_len)])
    with torch.no_grad():
        prob = model(seq).item()
    sentiment  = "Positive" if prob > 0.5 else "Negative"
    confidence = prob if prob > 0.5 else 1 - prob
    print(f"{review[:48]:<50} {sentiment:<12} {confidence*100:.1f}%")


print("\n===== WHAT I LEARNED TODAY =====")
print("Sequence Models - why order matters")
print("RNN - hidden state as memory")
print("Vanishing Gradient - why RNN fails long sequences")
print("LSTM - forget, input, output gates")
print("Cell State - long term memory highway")
print("GRU - simplified LSTM with 2 gates")
print("RNN Architectures - one to many, many to one")
print("Bidirectional LSTM - sees past and future")
print("Mini Project - Sentiment Analysis with LSTM")
print("\nDay 24 Done! Tomorrow - Attention Mechanism!")