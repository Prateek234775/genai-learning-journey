# ============================================
# DAY 32 - Fine Tuning LLMs
# PEFT, LoRA, QLoRA Concepts
# Author: Prateek Kumar Kuntal
# Date: 05 June 2026
# ============================================

import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


# ------------------------------------------
# PART 1 - WHY PEFT AND LORA
# ------------------------------------------

print("===== PART 1: Why PEFT and LoRA =====")

print("""
PROBLEM WITH FULL FINE TUNING:
    GPT-3  = 175 billion parameters
    Full fine tuning updates all 175B weights
    Requires massive GPU memory (thousands of GBs)
    Takes weeks on expensive hardware
    Storing fine tuned model = another 350GB file
    Cannot fine tune on consumer hardware

PEFT (Parameter Efficient Fine Tuning):
    Fine tune only a small fraction of parameters
    Keep most pretrained weights frozen
    Much less memory and compute needed
    Performance close to full fine tuning
    Multiple PEFT methods available

PEFT METHODS:
    LoRA      - most popular, what we use today
    Prefix Tuning  - add learnable prefix tokens
    Prompt Tuning  - learn soft prompt embeddings
    Adapter       - add small adapter layers
    IA3           - learn scaling vectors

LORA (Low Rank Adaptation):
    Introduced by Microsoft in 2021
    Adds small trainable matrices to attention layers
    Freezes all original model weights
    Only trains the small LoRA matrices
    Reduces trainable parameters by 10000x or more

    GPT-3 fine tuning:
    Full fine tuning : 175B trainable parameters
    LoRA fine tuning : 1-10M trainable parameters
    Same performance, fraction of the cost!
""")


# ------------------------------------------
# PART 2 - HOW LORA WORKS
# ------------------------------------------

print("===== PART 2: How LoRA Works =====")

print("""
LORA MATHEMATICS:
    Original weight matrix W has shape (d, k)
    Full fine tuning updates W directly
    W_new = W + delta_W

    LoRA instead learns low rank decomposition:
    delta_W = A @ B
    A has shape (d, r)
    B has shape (r, k)
    r is the rank, typically 4, 8, 16, 32

    Total parameters in delta_W:
    Full: d * k (example 768 * 768 = 589,824)
    LoRA: d*r + r*k = r*(d+k) (with r=8: 8*1536 = 12,288)
    Reduction: 589824 / 12288 = 48x fewer parameters!

WHY LOW RANK WORKS:
    Weight updates during fine tuning have low rank
    Task specific knowledge lives in low dimensional space
    High rank is redundant for most downstream tasks
    Low rank captures the essential changes

KEY HYPERPARAMETERS:
    r (rank)        - size of low rank matrices
                      lower = fewer params, less expressive
                      higher = more params, more expressive
                      typical values: 4, 8, 16, 32

    alpha           - scaling factor for LoRA weights
                      usually set to same as r or 2*r
                      lora_alpha / r = scaling multiplier

    target_modules  - which layers to apply LoRA
                      typically attention layers
                      q_proj, v_proj, k_proj, o_proj

    dropout         - regularization for LoRA layers
                      typically 0.05 to 0.1
""")

# Demonstrate LoRA math
print("LoRA Parameter Reduction Demo:")
print(f"{'Config':<30} {'Full Params':>15} {'LoRA Params':>15} {'Reduction':>12}")
print("-" * 75)

configs = [
    ("BERT base (768x768)", 768, 768),
    ("GPT-2 (768x768)",     768, 768),
    ("GPT-2 large (1280x1280)", 1280, 1280),
]

ranks = [4, 8, 16]
for name, d, k in configs:
    full_params = d * k
    for r in ranks:
        lora_params = r * (d + k)
        reduction   = full_params / lora_params
        print(f"{name+' r='+str(r):<30} {full_params:>15,} "
              f"{lora_params:>15,} {reduction:>11.1f}x")
    print()


# ------------------------------------------
# PART 3 - LORA FOR CLASSIFICATION
# ------------------------------------------

print("===== PART 3: LoRA for Classification =====")

# Dataset
positive = [
    "this restaurant has amazing food and great service",
    "absolutely loved the experience highly recommended",
    "best meal i have ever had fantastic atmosphere",
    "excellent food quality and very friendly staff",
    "wonderful dining experience will definitely return",
    "outstanding cuisine and impeccable service overall",
    "delicious food perfect ambiance loved every moment",
    "superb quality ingredients beautifully presented dishes",
    "incredible flavors and attentive knowledgeable staff",
    "perfect evening great food lovely atmosphere and service",
    "highly recommend this place food was extraordinary",
    "amazing experience the chef created culinary masterpiece",
]

negative = [
    "terrible food and extremely rude staff avoid this place",
    "worst restaurant experience ever complete disappointment",
    "food was cold and tasteless very poor service quality",
    "overpriced and underdelivered will never return again",
    "awful dining experience food arrived late and cold",
    "disgusting food and filthy place health hazard avoid",
    "extremely disappointing meal and horrible rude service",
    "bad food long wait times very poor value for money",
    "inedible food terrible service complete waste of money",
    "dreadful experience food was awful staff were rude",
    "very poor quality food and extremely slow service",
    "horrible meal cold food rude waiter never coming back",
]

texts  = positive + negative
labels = [1] * len(positive) + [0] * len(negative)

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print(f"Dataset prepared:")
print(f"  Total samples  : {len(texts)}")
print(f"  Train samples  : {len(X_train)}")
print(f"  Test samples   : {len(X_test)}")

# Load model
model_name = "distilbert-base-uncased"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label={0: "Negative", 1: "Positive"},
    label2id={"Negative": 0, "Positive": 1}
)

# Count base model parameters
total_base = sum(p.numel() for p in base_model.parameters())
print(f"\nBase model      : {model_name}")
print(f"Total parameters: {total_base:,}")


# ------------------------------------------
# PART 4 - APPLY LORA
# ------------------------------------------

print("\n===== PART 4: Apply LoRA =====")

# LoRA configuration
lora_config = LoraConfig(
    task_type       = TaskType.SEQ_CLS,
    r               = 8,           # rank
    lora_alpha      = 16,          # scaling
    lora_dropout    = 0.1,
    bias            = "none",
    target_modules  = ["q_lin", "v_lin"],  # DistilBERT attention
)

print("LoRA Configuration:")
print(f"  Rank (r)        : {lora_config.r}")
print(f"  Alpha           : {lora_config.lora_alpha}")
print(f"  Dropout         : {lora_config.lora_dropout}")
print(f"  Target modules  : {lora_config.target_modules}")
print(f"  Scaling factor  : {lora_config.lora_alpha / lora_config.r}")

# Apply LoRA to model
lora_model = get_peft_model(base_model, lora_config)

# Count parameters
total_params     = sum(p.numel() for p in lora_model.parameters())
trainable_params = sum(p.numel() for p in lora_model.parameters()
                       if p.requires_grad)
frozen_params    = total_params - trainable_params

print(f"\nParameter comparison:")
print(f"  Total parameters    : {total_params:,}")
print(f"  Trainable (LoRA)    : {trainable_params:,}")
print(f"  Frozen (pretrained) : {frozen_params:,}")
print(f"  Trainable %         : "
      f"{trainable_params/total_params*100:.3f}%")
print(f"  Parameter reduction : "
      f"{total_params/trainable_params:.1f}x")

# Print LoRA model summary
lora_model.print_trainable_parameters()


# ------------------------------------------
# PART 5 - TOKENIZE AND TRAIN
# ------------------------------------------

print("\n===== PART 5: Tokenize and Train =====")

def tokenize(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
        padding=False
    )

train_dataset = Dataset.from_dict(
    {"text": X_train, "label": y_train})
test_dataset  = Dataset.from_dict(
    {"text": X_test, "label": y_test})

train_tokenized = train_dataset.map(tokenize, batched=True)
test_tokenized  = test_dataset.map(tokenize, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions    = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1"      : f1_score(labels, predictions,
                              average="weighted"),
    }

training_args = TrainingArguments(
    output_dir                  = "lora_restaurant_classifier",
    num_train_epochs            = 5,
    per_device_train_batch_size = 4,
    per_device_eval_batch_size  = 4,
    learning_rate               = 3e-4,    # higher lr for LoRA
    weight_decay                = 0.01,
    evaluation_strategy         = "epoch",
    save_strategy               = "epoch",
    load_best_model_at_end      = True,
    metric_for_best_model       = "accuracy",
    report_to                   = "none",
    logging_steps               = 5,
    seed                        = 42,
)

trainer = Trainer(
    model           = lora_model,
    args            = training_args,
    train_dataset   = train_tokenized,
    eval_dataset    = test_tokenized,
    tokenizer       = tokenizer,
    data_collator   = data_collator,
    compute_metrics = compute_metrics,
)

print("Training LoRA model...")
print(f"Only {trainable_params:,} parameters being trained!")
print(f"{'='*60}")

train_result = trainer.train()

print(f"\nTraining completed!")
print(f"Training loss   : {train_result.training_loss:.4f}")


# ------------------------------------------
# PART 6 - EVALUATE LORA MODEL
# ------------------------------------------

print("\n===== PART 6: Evaluate LoRA Model =====")

eval_results = trainer.evaluate()
print(f"Evaluation Results:")
for key, value in eval_results.items():
    if isinstance(value, float):
        print(f"  {key:<30} : {value:.4f}")

# Predictions
predictions  = trainer.predict(test_tokenized)
pred_labels  = np.argmax(predictions.predictions, axis=-1)
true_labels  = predictions.label_ids

accuracy     = accuracy_score(true_labels, pred_labels)
f1           = f1_score(true_labels, pred_labels,
                         average="weighted")

print(f"\nFinal Results:")
print(f"  Accuracy        : {accuracy*100:.2f}%")
print(f"  F1 Score        : {f1:.4f}")

# Test new reviews
def predict_sentiment(text, model, tokenizer):
    model.eval()
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    )
    with torch.no_grad():
        outputs = model(**encoding)
        probs   = torch.softmax(outputs.logits, dim=-1)
        pred    = probs.argmax(dim=-1).item()

    labels = {0: "Negative", 1: "Positive"}
    return {
        "label"     : labels[pred],
        "confidence": probs[0][pred].item()
    }

new_reviews = [
    "absolutely fantastic food and wonderful service tonight",
    "terrible experience food was cold and staff were rude",
    "decent food nothing special but not bad either",
    "best restaurant in the city highly highly recommend",
    "waited 45 minutes for mediocre food very disappointed",
]

print(f"\nPredictions on new reviews:")
print(f"{'Review':<50} {'Sentiment':<12} {'Confidence'}")
print("-" * 72)
for review in new_reviews:
    result = predict_sentiment(review, lora_model, tokenizer)
    print(f"{review[:48]:<50} {result['label']:<12} "
          f"{result['confidence']*100:.1f}%")


# ------------------------------------------
# PART 7 - SAVE AND LOAD LORA WEIGHTS
# ------------------------------------------

print("\n===== PART 7: Save and Load LoRA Weights =====")

print("""
SAVING LORA MODELS:
    Only save LoRA adapter weights not full model
    LoRA weights are tiny compared to full model

    Full DistilBERT model  : ~260 MB
    LoRA adapter weights   : ~1-5 MB

    To use saved model:
    Load original base model
    Load LoRA adapter on top
    Ready for inference

    This is how HuggingFace Hub works for fine tuned models
    You upload only the adapter not the full model
""")

# Save LoRA adapter
save_path = "lora_adapter_restaurant"
lora_model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

import os
adapter_files = os.listdir(save_path)
print(f"Saved files in {save_path}:")
for f in adapter_files:
    size = os.path.getsize(os.path.join(save_path, f))
    print(f"  {f:<35} {size/1024:.1f} KB")

# Load LoRA adapter on base model
print(f"\nLoading LoRA adapter...")
loaded_base = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)
loaded_lora = PeftModel.from_pretrained(
    loaded_base, save_path)
loaded_tokenizer = AutoTokenizer.from_pretrained(save_path)

print(f"LoRA adapter loaded successfully!")

# Verify loaded model works
test_text = "amazing food and wonderful staff highly recommend"
result    = predict_sentiment(
    test_text, loaded_lora, loaded_tokenizer)
print(f"\nLoaded model prediction:")
print(f"Text       : {test_text}")
print(f"Prediction : {result['label']} "
      f"({result['confidence']*100:.1f}% confident)")


# ------------------------------------------
# PART 8 - QLORA CONCEPTS
# ------------------------------------------

print("\n===== PART 8: QLoRA Concepts =====")

print("""
QLORA (Quantized LoRA):
    Introduced by Dettmers et al. in 2023
    Combines quantization with LoRA
    Fine tune 65B parameter models on single GPU!

QUANTIZATION:
    Reduce precision of model weights
    float32 = 4 bytes per parameter (standard)
    float16 = 2 bytes per parameter (half precision)
    int8    = 1 byte per parameter
    int4    = 0.5 bytes per parameter (QLoRA uses this)

    65B model in float32 : 65B * 4 = 260 GB
    65B model in int4    : 65B * 0.5 = 32.5 GB
    Fits in single A100 GPU!

HOW QLORA WORKS:
    1. Quantize base model to 4-bit (NF4 format)
    2. Add LoRA adapters in float16
    3. Train only LoRA adapters
    4. Quantized base model stays frozen
    5. Gradients flow through quantized model to adapters

QLORA COMPONENTS:
    NF4 quantization     - special 4-bit format for weights
    Double quantization  - quantize the quantization constants
    Paged optimizers     - handle memory spikes efficiently
    bitsandbytes library - handles the quantization

MEMORY COMPARISON for LLaMA 7B:
    Full fine tuning fp32  : 112 GB
    Full fine tuning fp16  : 56 GB
    LoRA fp16              : 14 GB
    QLoRA 4-bit            : 5 GB  (fits on free Colab GPU!)

CODE FOR QLORA (requires GPU):
    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_quant_type       = "nf4",
        bnb_4bit_compute_dtype    = torch.float16,
        bnb_4bit_use_double_quant = True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        quantization_config = bnb_config,
        device_map          = "auto",
    )

    We cannot run this on CPU but you will use
    QLoRA on Google Colab in Day 33!
""")

# Simulate memory savings calculation
print("Memory Requirements Comparison:")
print(f"{'Method':<30} {'Precision':<12} {'Memory (7B model)'}")
print("-" * 58)

model_params = 7e9  # 7 billion
configs_mem  = [
    ("Full fine tuning",  "float32", 4),
    ("Full fine tuning",  "float16", 2),
    ("LoRA fine tuning",  "float16", 2),
    ("QLoRA fine tuning", "int4",    0.5),
]

for method, precision, bytes_per in configs_mem:
    memory_gb = (model_params * bytes_per) / 1e9
    if "LoRA" in method and "Q" not in method:
        memory_gb *= 0.25   # LoRA is roughly 25% of full
    print(f"{method:<30} {precision:<12} {memory_gb:.1f} GB")


# ------------------------------------------
# PART 9 - LORA RANK COMPARISON
# ------------------------------------------

print("\n===== PART 9: LoRA Rank Comparison =====")

print("""
CHOOSING RANK (r):
    r=1  to r=4   - very few params, works for simple tasks
    r=8  to r=16  - good balance, works for most tasks
    r=32 to r=64  - more expressive, use for complex tasks
    r=128+        - approaching full fine tuning

RULE OF THUMB:
    Start with r=8
    If underfitting increase rank
    If overfitting decrease rank or increase dropout

ALPHA RELATIONSHIP:
    lora_alpha / r = effective scaling
    keeping alpha = 2*r is common practice
    alpha=16, r=8 gives scaling of 2
""")

print("Rank vs Parameter count (for 768x768 layer):")
print(f"{'Rank (r)':<12} {'Alpha':<10} {'LoRA Params':<15} "
      f"{'Full Params':<15} {'Reduction'}")
print("-" * 65)

d, k          = 768, 768
full_params   = d * k

for r in [1, 2, 4, 8, 16, 32, 64]:
    lora_params = r * (d + k)
    alpha       = r * 2
    reduction   = full_params / lora_params
    print(f"{r:<12} {alpha:<10} {lora_params:<15,} "
          f"{full_params:<15,} {reduction:.1f}x")


# ------------------------------------------
# MINI PROJECT - LoRA vs Full Fine Tuning
# ------------------------------------------

print("\n===== MINI PROJECT: LoRA vs Full Fine Tuning Comparison =====")

print("""
Comparing three approaches:
    1. Zero shot  - no fine tuning at all
    2. Full fine tune - update all parameters
    3. LoRA fine tune - update only LoRA params

On same dataset to show tradeoffs
""")

# Same dataset used above
# Zero shot baseline using pipeline
from transformers import pipeline

print("Evaluating zero shot baseline...")
zero_shot_pipe = pipeline(
    "text-classification",
    model  = "distilbert-base-uncased-finetuned-sst-2-english",
    device = -1
)

zero_shot_preds = []
for text in X_test:
    result = zero_shot_pipe(text)[0]
    pred   = 1 if result["label"] == "POSITIVE" else 0
    zero_shot_preds.append(pred)

zero_shot_acc = accuracy_score(y_test, zero_shot_preds)
print(f"Zero shot accuracy  : {zero_shot_acc*100:.2f}%")

# Full fine tuning
print(f"\nFull fine tuning baseline...")
full_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

full_args = TrainingArguments(
    output_dir                  = "full_finetune_compare",
    num_train_epochs            = 5,
    per_device_train_batch_size = 4,
    learning_rate               = 2e-5,
    evaluation_strategy         = "epoch",
    save_strategy               = "no",
    report_to                   = "none",
    seed                        = 42,
)

full_trainer = Trainer(
    model           = full_model,
    args            = full_args,
    train_dataset   = train_tokenized,
    eval_dataset    = test_tokenized,
    tokenizer       = tokenizer,
    data_collator   = data_collator,
    compute_metrics = compute_metrics,
)

full_trainer.train()
full_preds      = full_trainer.predict(test_tokenized)
full_pred_labels= np.argmax(full_preds.predictions, axis=-1)
full_acc        = accuracy_score(y_test, full_pred_labels)

full_trainable  = sum(
    p.numel() for p in full_model.parameters())

print(f"Full fine tune accuracy: {full_acc*100:.2f}%")

# Summary comparison
print(f"\nFinal Comparison Summary:")
print(f"{'Method':<25} {'Accuracy':>12} {'Params Trained':>18} {'Relative Cost'}")
print("-" * 72)

lora_acc = accuracy_score(true_labels, pred_labels)

print(f"{'Zero Shot':<25} {zero_shot_acc*100:>11.2f}% "
      f"{'0':>18} {'Free'}")
print(f"{'LoRA Fine Tuning':<25} {lora_acc*100:>11.2f}% "
      f"{trainable_params:>18,} {'Very Low'}")
print(f"{'Full Fine Tuning':<25} {full_acc*100:>11.2f}% "
      f"{full_trainable:>18,} {'High'}")

print(f"""
Key Takeaways:
    LoRA achieves similar accuracy to full fine tuning
    LoRA trains {full_trainable/trainable_params:.0f}x fewer parameters
    LoRA is the industry standard for LLM fine tuning
    QLoRA extends this to billion parameter models
""")


print("\n===== WHAT I LEARNED TODAY =====")
print("Why full fine tuning is impractical for large models")
print("PEFT - parameter efficient fine tuning overview")
print("LoRA - low rank adaptation mathematics")
print("Rank and alpha hyperparameters")
print("Applying LoRA with HuggingFace PEFT library")
print("Training with only 1 percent of parameters")
print("Saving and loading LoRA adapter weights")
print("QLoRA - quantized LoRA for billion param models")
print("Comparison of zero shot vs LoRA vs full fine tuning")
print("\nDay 32 Done! Tomorrow - Fine Tune LLM on Custom Dataset!")