# ============================================
# DAY 30 - Text Classification using BERT
# Fine tuning with HuggingFace Trainer
# Author: Prateek Kumar Kuntal
# Date: 03 June 2026
# ============================================

import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


# ------------------------------------------
# PART 1 - WHY FINE TUNE BERT
# ------------------------------------------

print("===== PART 1: Why Fine Tune BERT =====")

print("""
WHY FINE TUNE INSTEAD OF TRAIN FROM SCRATCH:
    BERT already knows English language deeply
    Pre-trained on 3.3 billion words
    Fine tuning just adapts to your specific task
    Need very little labeled data (hundreds to thousands)
    Training time: minutes instead of days

WHAT CHANGES DURING FINE TUNING:
    All BERT weights are updated slightly
    New classification head added and trained
    Learning rate very small to not destroy pretrained weights
    Typical lr: 2e-5 to 5e-5

WHAT STAYS SAME:
    BERT architecture unchanged
    Tokenizer unchanged
    Core language understanding preserved

DISTILBERT vs BERT:
    DistilBERT = smaller faster version of BERT
    40% smaller, 60% faster, 97% of BERT performance
    Great for when speed matters
    We use DistilBERT today for faster training
""")


# ------------------------------------------
# PART 2 - PREPARE DATASET
# ------------------------------------------

print("===== PART 2: Prepare Dataset =====")

# Multi class news classification dataset
tech_news = [
    "Apple launches new iPhone with advanced AI camera features",
    "Google releases Gemini ultra the most powerful AI model",
    "Microsoft integrates ChatGPT into all office products",
    "Meta releases Llama open source large language model",
    "OpenAI announces GPT-5 with improved reasoning capabilities",
    "NVIDIA GPU chips power most of the AI revolution today",
    "Python remains the most popular programming language for AI",
    "HuggingFace reaches 500000 models on their platform",
    "Tesla autopilot uses deep learning for self driving cars",
    "Amazon AWS launches new machine learning services for developers",
    "Samsung releases new chip designed specifically for AI workloads",
    "Intel announces new processor optimized for neural networks",
    "Anthropic releases Claude AI assistant with safety features",
    "DeepMind achieves breakthrough in protein structure prediction",
    "GitHub Copilot helps millions of developers write code faster",
]

sports_news = [
    "India wins the cricket world cup in a thrilling final match",
    "Virat Kohli scores century in the test match against Australia",
    "Rohit Sharma leads team India to victory in the series",
    "Manchester United signs new striker for record transfer fee",
    "Real Madrid wins UEFA Champions League for record 15th time",
    "Lionel Messi announces retirement from international football",
    "Serena Williams returns to tennis after injury comeback",
    "Roger Federer wins his 21st grand slam title at Wimbledon",
    "Olympic games 2024 Paris sees record participation worldwide",
    "Neeraj Chopra wins gold medal at world athletics championship",
    "IPL 2024 sees highest viewership in tournament history",
    "Novak Djokovic wins Australian Open for record 25th slam",
    "Formula 1 season sees fierce battle between top drivers",
    "NBA finals goes to game 7 in the most exciting series ever",
    "PV Sindhu wins silver medal at the badminton world championship",
]

business_news = [
    "Stock market reaches all time high driven by tech stocks",
    "Inflation rate drops to lowest level in three years globally",
    "Reserve Bank of India keeps interest rates unchanged today",
    "Startup ecosystem in India raises record funding this quarter",
    "Tata Group announces major investment in semiconductor chip",
    "Reliance Industries reports strong quarterly earnings growth",
    "Foreign investment in India increases significantly this year",
    "Rupee strengthens against dollar after positive economic data",
    "Global oil prices rise due to supply concerns in Middle East",
    "Amazon reports strong revenue growth in quarterly results",
    "Goldman Sachs predicts strong GDP growth for India next year",
    "Venture capital funding for AI startups reaches record high",
    "Infosys and TCS win large contracts from global companies",
    "Electric vehicle sales double in India compared to last year",
    "Bitcoin price surges past 70000 dollars hitting new record",
]

texts  = tech_news + sports_news + business_news
labels = [0] * len(tech_news) + [1] * len(sports_news) + [2] * len(business_news)

label_names = ["Technology", "Sports", "Business"]
id2label    = {0: "Technology", 1: "Sports", 2: "Business"}
label2id    = {"Technology": 0, "Sports": 1, "Business": 2}

print(f"Total samples    : {len(texts)}")
print(f"Technology       : {labels.count(0)}")
print(f"Sports           : {labels.count(1)}")
print(f"Business         : {labels.count(2)}")

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print(f"\nTrain samples    : {len(X_train)}")
print(f"Test samples     : {len(X_test)}")


# ------------------------------------------
# PART 3 - TOKENIZATION
# ------------------------------------------

print("\n===== PART 3: Tokenization =====")

model_name = "distilbert-base-uncased"
tokenizer  = AutoTokenizer.from_pretrained(model_name)

print(f"Model            : {model_name}")
print(f"Vocab size       : {tokenizer.vocab_size}")
print(f"Max length       : {tokenizer.model_max_length}")

# Show tokenization example
sample_text = texts[0]
tokens      = tokenizer.tokenize(sample_text)
encoding    = tokenizer(sample_text)

print(f"\nSample text      : {sample_text}")
print(f"Tokens           : {tokens}")
print(f"Token IDs        : {encoding['input_ids']}")
print(f"Num tokens       : {len(tokens)}")

# Tokenize full dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
        padding=False      # DataCollator handles padding
    )

# Create HuggingFace datasets
train_dict = {"text": X_train, "label": y_train}
test_dict  = {"text": X_test,  "label": y_test}

train_dataset = Dataset.from_dict(train_dict)
test_dataset  = Dataset.from_dict(test_dict)

print(f"\nDataset created:")
print(f"Train dataset    : {train_dataset}")
print(f"Test dataset     : {test_dataset}")

# Tokenize datasets
train_tokenized = train_dataset.map(
    tokenize_function, batched=True)
test_tokenized  = test_dataset.map(
    tokenize_function, batched=True)

print(f"\nTokenized train  : {train_tokenized}")
print(f"Columns          : {train_tokenized.column_names}")


# ------------------------------------------
# PART 4 - LOAD PRETRAINED MODEL
# ------------------------------------------

print("\n===== PART 4: Load Pretrained Model =====")

print(f"Loading {model_name} for sequence classification...")

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
    id2label=id2label,
    label2id=label2id
)

total_params    = sum(p.numel() for p in model.parameters())
trainable_params= sum(p.numel() for p in model.parameters()
                      if p.requires_grad)

print(f"Total parameters    : {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"\nModel architecture:")
print(model.config)


# ------------------------------------------
# PART 5 - TRAINING SETUP
# ------------------------------------------

print("\n===== PART 5: Training Setup =====")

print("""
TRAINING ARGUMENTS:
    output_dir      - where to save checkpoints
    num_epochs      - how many times to see data
    per_device_batch- batch size per GPU/CPU
    learning_rate   - how fast to update weights
    weight_decay    - L2 regularization
    evaluation_strategy - when to evaluate
    save_strategy   - when to save checkpoints
    load_best_model - load best checkpoint at end

BERT FINE TUNING TIPS:
    Learning rate  : 2e-5 to 5e-5 (very important)
    Batch size     : 16 or 32
    Epochs         : 3 to 5 (more can overfit)
    Warmup steps   : 10% of total steps
    Weight decay   : 0.01
""")

# Data collator handles dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions    = np.argmax(logits, axis=-1)
    accuracy       = accuracy_score(labels, predictions)
    f1             = f1_score(labels, predictions,
                               average="weighted")
    return {
        "accuracy": accuracy,
        "f1"      : f1,
    }

# Training arguments
training_args = TrainingArguments(
    output_dir                  = "bert_news_classifier",
    num_train_epochs            = 5,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size  = 8,
    learning_rate               = 2e-5,
    weight_decay                = 0.01,
    warmup_ratio                = 0.1,
    evaluation_strategy         = "epoch",
    save_strategy               = "epoch",
    load_best_model_at_end      = True,
    metric_for_best_model       = "accuracy",
    logging_steps               = 10,
    report_to                   = "none",
    seed                        = 42,
)

print("Training Arguments:")
print(f"  Epochs           : {training_args.num_train_epochs}")
print(f"  Batch size       : {training_args.per_device_train_batch_size}")
print(f"  Learning rate    : {training_args.learning_rate}")
print(f"  Weight decay     : {training_args.weight_decay}")
print(f"  Warmup ratio     : {training_args.warmup_ratio}")


# ------------------------------------------
# PART 6 - TRAIN WITH TRAINER API
# ------------------------------------------

print("\n===== PART 6: Train with Trainer API =====")

trainer = Trainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_tokenized,
    eval_dataset    = test_tokenized,
    tokenizer       = tokenizer,
    data_collator   = data_collator,
    compute_metrics = compute_metrics,
)

print("Starting training...")
print(f"{'='*60}")

train_result = trainer.train()

print(f"\nTraining completed!")
print(f"Training loss    : {train_result.training_loss:.4f}")
print(f"Training time    : {train_result.metrics['train_runtime']:.2f}s")


# ------------------------------------------
# PART 7 - EVALUATE MODEL
# ------------------------------------------

print("\n===== PART 7: Evaluate Model =====")

# Evaluate on test set
eval_results = trainer.evaluate()

print(f"Evaluation Results:")
for key, value in eval_results.items():
    if isinstance(value, float):
        print(f"  {key:<30} : {value:.4f}")

# Get predictions
predictions    = trainer.predict(test_tokenized)
pred_labels    = np.argmax(predictions.predictions, axis=-1)
true_labels    = predictions.label_ids

print(f"\nDetailed Classification Report:")
print(classification_report(
    true_labels, pred_labels,
    target_names=label_names
))

print(f"Confusion Matrix:")
cm = confusion_matrix(true_labels, pred_labels)
print(f"{'':>12}", end="")
for name in label_names:
    print(f"{name[:8]:>10}", end="")
print()
for i, name in enumerate(label_names):
    print(f"{name[:10]:>12}", end="")
    for j in range(len(label_names)):
        print(f"{cm[i][j]:>10}", end="")
    print()


# ------------------------------------------
# PART 8 - INFERENCE ON NEW TEXTS
# ------------------------------------------

print("\n===== PART 8: Inference on New Texts =====")

def predict_category(text, model, tokenizer,
                     id2label, device="cpu"):
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
        logits  = outputs.logits
        probs   = torch.softmax(logits, dim=-1)
        pred_id = probs.argmax(dim=-1).item()

    return {
        "label"      : id2label[pred_id],
        "confidence" : probs[0][pred_id].item(),
        "all_scores" : {
            id2label[i]: probs[0][i].item()
            for i in range(len(id2label))
        }
    }

new_texts = [
    "Prateek Kumar wins gold medal at national coding championship",
    "RBI announces new digital payment infrastructure for India",
    "OpenAI releases new model that can reason step by step",
    "Mumbai Indians win IPL trophy defeating Chennai Super Kings",
    "Sensex crosses 80000 points for first time in history",
    "Meta AI announces new open source vision language model",
]

print(f"Predictions on new texts:")
print(f"{'Text':<55} {'Label':<12} {'Confidence'}")
print("-" * 80)

for text in new_texts:
    result = predict_category(
        text, model, tokenizer, id2label)
    print(f"{text[:53]:<55} {result['label']:<12} "
          f"{result['confidence']*100:.1f}%")

# Detailed prediction for one text
print(f"\nDetailed prediction example:")
text   = "Google DeepMind releases AlphaFold 3 for drug discovery"
result = predict_category(text, model, tokenizer, id2label)
print(f"Text   : {text}")
print(f"Prediction: {result['label']} "
      f"({result['confidence']*100:.1f}% confident)")
print(f"\nAll category scores:")
for category, score in result["all_scores"].items():
    bar = "#" * int(score * 40)
    print(f"  {category:<12} : {bar:<40} {score*100:.1f}%")


# ------------------------------------------
# PART 9 - SAVE AND LOAD MODEL
# ------------------------------------------

print("\n===== PART 9: Save and Load Model =====")

# Save model and tokenizer
save_path = "saved_bert_classifier"
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)
print(f"Model saved to   : {save_path}/")

# Load model back
print(f"Loading model from {save_path}...")
loaded_tokenizer = AutoTokenizer.from_pretrained(save_path)
loaded_model     = AutoModelForSequenceClassification.from_pretrained(
    save_path)

# Verify loaded model works
test_text   = "India beats Australia in cricket world cup final"
result      = predict_category(
    test_text, loaded_model,
    loaded_tokenizer, id2label)

print(f"\nLoaded model prediction:")
print(f"Text      : {test_text}")
print(f"Prediction: {result['label']} "
      f"({result['confidence']*100:.1f}% confident)")
print(f"Model loaded and working correctly!")


# ------------------------------------------
# MINI PROJECT - News Category API
# ------------------------------------------

print("\n===== MINI PROJECT: News Category Classifier =====")

class NewsCategoryClassifier:
    def __init__(self, model, tokenizer, id2label):
        self.model     = model
        self.tokenizer = tokenizer
        self.id2label  = id2label
        self.model.eval()

    def classify(self, text):
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        )
        with torch.no_grad():
            outputs = self.model(**encoding)
            probs   = torch.softmax(
                outputs.logits, dim=-1)[0]

        results = {
            self.id2label[i]: probs[i].item()
            for i in range(len(self.id2label))
        }
        best_label = max(results, key=results.get)

        return {
            "category"   : best_label,
            "confidence" : results[best_label],
            "scores"     : results
        }

    def classify_batch(self, texts):
        return [self.classify(text) for text in texts]

    def get_top_category(self, text):
        result = self.classify(text)
        return (f"{result['category']} "
                f"({result['confidence']*100:.1f}%)")

classifier = NewsCategoryClassifier(
    loaded_model, loaded_tokenizer, id2label)

batch_texts = [
    "Nifty 50 index rises 2 percent on strong earnings",
    "Djokovic wins French Open defeating Carlos Alcaraz",
    "Qualcomm releases new Snapdragon chip for AI phones",
    "Petrol and diesel prices remain stable this month",
    "Hardik Pandya takes five wickets against West Indies",
    "GitHub launches AI powered code review feature",
]

print(f"Batch Classification Results:")
print(f"{'Text':<50} {'Result'}")
print("-" * 75)

for text in batch_texts:
    category = classifier.get_top_category(text)
    print(f"{text[:48]:<50} {category}")


print("\n===== WHAT I LEARNED TODAY =====")
print("Why fine tune BERT instead of training from scratch")
print("Preparing dataset for HuggingFace Trainer")
print("Tokenizing with AutoTokenizer")
print("Loading pretrained DistilBERT for classification")
print("TrainingArguments and Trainer API")
print("Evaluation with accuracy and F1 score")
print("Confusion matrix and classification report")
print("Inference on new texts with confidence scores")
print("Save and load fine tuned model")
print("Mini Project - News Category Classifier class")
print("\nDay 30 Done! Tomorrow - Text Generation with GPT-2!")