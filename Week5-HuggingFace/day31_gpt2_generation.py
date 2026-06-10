# ============================================
# DAY 31 - Text Generation with GPT-2
# Author: Prateek Kumar Kuntal
# Date: 04 June 2026
# ============================================

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    GPT2Tokenizer,
    GPT2LMHeadModel,
)
import numpy as np


# ------------------------------------------
# PART 1 - WHAT IS TEXT GENERATION
# ------------------------------------------

print("===== PART 1: What is Text Generation =====")

print("""
TEXT GENERATION:
    Model predicts next token given previous tokens
    Repeat until reaching max length or stop token
    This is how ChatGPT, Claude, Gemini work

    Input  : "Machine learning is"
    Step 1 : "Machine learning is a"
    Step 2 : "Machine learning is a subset"
    Step 3 : "Machine learning is a subset of"
    Step 4 : "Machine learning is a subset of AI"

GPT-2 MODEL:
    Released by OpenAI in 2019
    117M to 1.5B parameters (4 sizes)
    Trained on 40GB of internet text
    Generates surprisingly coherent text
    Fully open source on HuggingFace

GPT-2 SIZES:
    gpt2          - 117M parameters  (we use this)
    gpt2-medium   - 345M parameters
    gpt2-large    - 774M parameters
    gpt2-xl       - 1.5B parameters

GENERATION STRATEGIES:
    Greedy search      - always pick highest probability
    Beam search        - explore multiple paths
    Top-k sampling     - sample from top k tokens
    Top-p sampling     - sample from top p probability mass
    Temperature        - control randomness
""")


# ------------------------------------------
# PART 2 - LOAD GPT-2
# ------------------------------------------

print("===== PART 2: Load GPT-2 =====")

print("Loading GPT-2 model and tokenizer...")

model_name = "gpt2"
tokenizer  = GPT2Tokenizer.from_pretrained(model_name)
model      = GPT2LMHeadModel.from_pretrained(model_name)

# GPT-2 uses EOS as padding token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

total_params = sum(p.numel() for p in model.parameters())
print(f"Model            : {model_name}")
print(f"Parameters       : {total_params:,}")
print(f"Vocab size       : {tokenizer.vocab_size}")
print(f"Max length       : {tokenizer.model_max_length}")
print(f"Device           : {'cuda' if torch.cuda.is_available() else 'cpu'}")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")
model  = model.to(device)
model.eval()

# Quick tokenization example
text    = "Artificial intelligence is transforming"
encoded = tokenizer.encode(text, return_tensors="pt")
print(f"\nSample tokenization:")
print(f"Text    : {text}")
print(f"Tokens  : {tokenizer.tokenize(text)}")
print(f"IDs     : {encoded[0].tolist()}")
print(f"Length  : {len(encoded[0])}")


# ------------------------------------------
# PART 3 - GREEDY SEARCH
# ------------------------------------------

print("\n===== PART 3: Greedy Search =====")

print("""
GREEDY SEARCH:
    At each step pick token with highest probability
    Deterministic - same output every run
    Fast but often repetitive and boring
    Not recommended for creative text

    Step 1: pick argmax(probabilities)
    Step 2: append to input
    Step 3: repeat
""")

def generate_greedy(prompt, model, tokenizer,
                    max_new_tokens=50):
    inputs = tokenizer.encode(
        prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens   = max_new_tokens,
            do_sample        = False,   # greedy
            pad_token_id     = tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        outputs[0], skip_special_tokens=True)
    return generated

prompts = [
    "Machine learning is a",
    "The future of artificial intelligence",
    "Python is the best programming language for",
]

print("Greedy Search Generation:")
for prompt in prompts:
    generated = generate_greedy(prompt, model, tokenizer)
    print(f"\nPrompt   : {prompt}")
    print(f"Generated: {generated}")


# ------------------------------------------
# PART 4 - BEAM SEARCH
# ------------------------------------------

print("\n===== PART 4: Beam Search =====")

print("""
BEAM SEARCH:
    Keeps top-k sequences (beams) at each step
    Explores multiple paths simultaneously
    Picks sequence with highest overall probability
    Better than greedy but still can be repetitive

    num_beams=4 means track 4 best sequences
    At each step expand all beams, keep top 4
    Return best complete sequence at end

PARAMETERS:
    num_beams           - number of beams (4-8 typical)
    early_stopping      - stop when all beams hit EOS
    no_repeat_ngram_size- prevent repeating n-grams
    length_penalty      - penalize short/long sequences
""")

def generate_beam(prompt, model, tokenizer,
                  max_new_tokens=60, num_beams=4):
    inputs = tokenizer.encode(
        prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens       = max_new_tokens,
            num_beams            = num_beams,
            early_stopping       = True,
            no_repeat_ngram_size = 2,
            length_penalty       = 1.0,
            pad_token_id         = tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        outputs[0], skip_special_tokens=True)
    return generated

print("Beam Search Generation (num_beams=4):")
for prompt in prompts:
    generated = generate_beam(prompt, model, tokenizer)
    print(f"\nPrompt   : {prompt}")
    print(f"Generated: {generated}")


# ------------------------------------------
# PART 5 - SAMPLING STRATEGIES
# ------------------------------------------

print("\n===== PART 5: Sampling Strategies =====")

print("""
TEMPERATURE SAMPLING:
    Divide logits by temperature before softmax
    Temperature < 1 - sharper distribution, more focused
    Temperature > 1 - flatter distribution, more random
    Temperature = 1 - use model distribution as is

TOP-K SAMPLING:
    Only sample from top k most likely tokens
    Prevents very unlikely tokens being chosen
    k=50 is common default value
    More focused than pure random sampling

TOP-P SAMPLING (Nucleus Sampling):
    Sample from smallest set of tokens
    whose cumulative probability >= p
    p=0.9 means top 90% probability mass
    More dynamic than top-k
    Preferred by most practitioners today

COMBINING STRATEGIES:
    Most models use top-p + temperature together
    temperature=0.7, top_p=0.9 is common default
    Balances creativity and coherence
""")

def generate_sample(prompt, model, tokenizer,
                    max_new_tokens=80,
                    temperature=1.0,
                    top_k=50,
                    top_p=0.95):
    inputs = tokenizer.encode(
        prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens   = max_new_tokens,
            do_sample        = True,
            temperature      = temperature,
            top_k            = top_k,
            top_p            = top_p,
            pad_token_id     = tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        outputs[0], skip_special_tokens=True)
    return generated

prompt = "Deep learning has revolutionized"

print("Effect of Temperature:")
print(f"Prompt: {prompt}\n")

for temp in [0.3, 0.7, 1.0, 1.5]:
    torch.manual_seed(42)
    generated = generate_sample(
        prompt, model, tokenizer,
        max_new_tokens=40,
        temperature=temp,
        top_k=50,
        top_p=0.95
    )
    print(f"Temperature={temp}:")
    print(f"  {generated}")
    print()

print("Effect of Top-K:")
print(f"Prompt: {prompt}\n")

for k in [5, 20, 50, 200]:
    torch.manual_seed(42)
    generated = generate_sample(
        prompt, model, tokenizer,
        max_new_tokens=40,
        temperature=0.7,
        top_k=k,
        top_p=1.0
    )
    print(f"Top-k={k}:")
    print(f"  {generated}")
    print()

print("Effect of Top-P:")
print(f"Prompt: {prompt}\n")

for p in [0.5, 0.7, 0.9, 0.99]:
    torch.manual_seed(42)
    generated = generate_sample(
        prompt, model, tokenizer,
        max_new_tokens=40,
        temperature=0.7,
        top_k=0,     # disable top-k
        top_p=p
    )
    print(f"Top-p={p}:")
    print(f"  {generated}")
    print()


# ------------------------------------------
# PART 6 - REPETITION PENALTY
# ------------------------------------------

print("===== PART 6: Repetition Penalty =====")

print("""
REPETITION PENALTY:
    Penalizes tokens that have already appeared
    Prevents model from repeating same phrases
    Value > 1 reduces repetition
    Value = 1 means no penalty
    Value 1.2 to 1.5 works well in practice

WITHOUT PENALTY:
    "The cat sat on the mat the cat sat on the mat..."

WITH PENALTY:
    "The cat sat on the mat and enjoyed the sunshine..."
""")

def generate_with_penalty(prompt, model, tokenizer,
                           max_new_tokens=80,
                           repetition_penalty=1.0):
    inputs = tokenizer.encode(
        prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens      = max_new_tokens,
            do_sample           = True,
            temperature         = 0.7,
            top_p               = 0.9,
            repetition_penalty  = repetition_penalty,
            pad_token_id        = tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        outputs[0], skip_special_tokens=True)
    return generated

prompt = "Neural networks learn from data by"

print(f"Prompt: {prompt}\n")
for penalty in [1.0, 1.2, 1.5, 2.0]:
    torch.manual_seed(42)
    generated = generate_with_penalty(
        prompt, model, tokenizer,
        max_new_tokens=60,
        repetition_penalty=penalty
    )
    print(f"Repetition penalty={penalty}:")
    print(f"  {generated}")
    print()


# ------------------------------------------
# PART 7 - MULTIPLE SEQUENCES
# ------------------------------------------

print("===== PART 7: Multiple Sequences =====")

print("""
GENERATING MULTIPLE SEQUENCES:
    num_return_sequences - generate N different outputs
    Must use sampling (do_sample=True)
    Useful for:
        Getting diverse responses
        Picking best from multiple options
        Data augmentation
""")

def generate_multiple(prompt, model, tokenizer,
                      max_new_tokens=50, n=3):
    inputs = tokenizer.encode(
        prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens       = max_new_tokens,
            do_sample            = True,
            temperature          = 0.8,
            top_p                = 0.9,
            num_return_sequences = n,
            pad_token_id         = tokenizer.eos_token_id,
        )

    sequences = [
        tokenizer.decode(out, skip_special_tokens=True)
        for out in outputs
    ]
    return sequences

prompt    = "The best way to learn machine learning is"
sequences = generate_multiple(
    prompt, model, tokenizer,
    max_new_tokens=50, n=3)

print(f"Prompt: {prompt}")
print(f"\nGenerating 3 different continuations:")
for i, seq in enumerate(sequences):
    print(f"\nOption {i+1}:")
    print(f"  {seq}")


# ------------------------------------------
# PART 8 - PIPELINE FOR GENERATION
# ------------------------------------------

print("\n===== PART 8: Pipeline for Generation =====")

print("Using HuggingFace pipeline for generation...")

gen_pipeline = pipeline(
    "text-generation",
    model     = model,
    tokenizer = tokenizer,
    device    = -1
)

prompts = [
    "Artificial intelligence will change",
    "The most important skill for a software engineer",
    "In the next 10 years machine learning will",
]

print("Pipeline Generation Results:")
for prompt in prompts:
    result = gen_pipeline(
        prompt,
        max_new_tokens       = 50,
        do_sample            = True,
        temperature          = 0.7,
        top_p                = 0.9,
        repetition_penalty   = 1.2,
        num_return_sequences = 1,
    )
    generated = result[0]["generated_text"]
    print(f"\nPrompt   : {prompt}")
    print(f"Generated: {generated}")


# ------------------------------------------
# PART 9 - PERPLEXITY
# ------------------------------------------

print("\n===== PART 9: Perplexity =====")

print("""
PERPLEXITY:
    Metric to evaluate language model quality
    Measures how surprised the model is by text
    Lower perplexity = better language model
    
    Perplexity = exp(average cross entropy loss)

    Low perplexity  = model finds text predictable
    High perplexity = model finds text surprising

    Used to compare different language models
    GPT-4 has much lower perplexity than GPT-2
    on standard benchmarks
""")

def compute_perplexity(text, model, tokenizer):
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    input_ids = encoding["input_ids"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss    = outputs.loss

    perplexity = torch.exp(loss).item()
    return perplexity

test_texts = [
    "Machine learning is a subset of artificial intelligence.",
    "The transformer architecture uses attention mechanisms.",
    "xkzq wprm lkjh zxcv qwer asdf mnbv.",  # random chars
    "Python is a popular programming language for data science.",
    "Asjdfk lkjsdf kjhsdf kjhsdf lkjhsdf.",  # gibberish
]

print("Perplexity comparison:")
print(f"{'Text':<55} {'Perplexity'}")
print("-" * 70)
for text in test_texts:
    ppl = compute_perplexity(text, model, tokenizer)
    print(f"{text[:53]:<55} {ppl:.2f}")

print("\nLower perplexity = more natural text for GPT-2")


# ------------------------------------------
# MINI PROJECT - AI Story Generator
# ------------------------------------------

print("\n===== MINI PROJECT: AI Story Generator =====")

class StoryGenerator:
    def __init__(self, model, tokenizer):
        self.model     = model
        self.tokenizer = tokenizer
        self.model.eval()

    def generate(self, prompt, style="balanced",
                 max_new_tokens=150):
        styles = {
            "focused"  : {"temperature": 0.5,
                          "top_p": 0.8,
                          "top_k": 30},
            "balanced" : {"temperature": 0.7,
                          "top_p": 0.9,
                          "top_k": 50},
            "creative" : {"temperature": 1.0,
                          "top_p": 0.95,
                          "top_k": 100},
        }

        params = styles.get(style, styles["balanced"])
        inputs = self.tokenizer.encode(
            prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens      = max_new_tokens,
                do_sample           = True,
                temperature         = params["temperature"],
                top_p               = params["top_p"],
                top_k               = params["top_k"],
                repetition_penalty  = 1.3,
                pad_token_id        = self.tokenizer.eos_token_id,
            )

        story = self.tokenizer.decode(
            outputs[0], skip_special_tokens=True)
        return story

    def generate_variations(self, prompt, n=3):
        variations = []
        for _ in range(n):
            story = self.generate(prompt, style="balanced",
                                  max_new_tokens=80)
            variations.append(story)
        return variations

    def get_stats(self, text):
        words    = len(text.split())
        sentences= text.count(".") + text.count("!") + text.count("?")
        ppl      = compute_perplexity(
            text, self.model, self.tokenizer)
        return {
            "words"      : words,
            "sentences"  : max(sentences, 1),
            "perplexity" : ppl,
        }

generator = StoryGenerator(model, tokenizer)

story_prompts = [
    "Once upon a time in a world powered by artificial intelligence",
    "The young programmer sat down and opened their laptop",
    "In 2050 machine learning had transformed every industry",
]

print("Story Generator Demo:")
print("=" * 70)

for prompt in story_prompts:
    print(f"\nPrompt: {prompt}")
    print(f"\nFocused style (temperature=0.5):")
    story = generator.generate(
        prompt, style="focused", max_new_tokens=80)
    stats = generator.get_stats(story)
    print(f"  {story}")
    print(f"  Words: {stats['words']} | "
          f"Perplexity: {stats['perplexity']:.2f}")

    print(f"\nCreative style (temperature=1.0):")
    torch.manual_seed(99)
    story = generator.generate(
        prompt, style="creative", max_new_tokens=80)
    stats = generator.get_stats(story)
    print(f"  {story}")
    print(f"  Words: {stats['words']} | "
          f"Perplexity: {stats['perplexity']:.2f}")
    print("-" * 70)


print("\n===== WHAT I LEARNED TODAY =====")
print("GPT-2 architecture and loading from HuggingFace")
print("Greedy search - deterministic but repetitive")
print("Beam search - better quality but still focused")
print("Temperature - controls randomness of generation")
print("Top-k sampling - sample from top k tokens")
print("Top-p sampling - nucleus sampling most popular")
print("Repetition penalty - prevent repetitive output")
print("Multiple sequence generation")
print("Perplexity - how to evaluate language models")
print("Mini Project - AI Story Generator with styles")
print("\nDay 31 Done! Tomorrow - Fine Tuning LLMs with LoRA!")