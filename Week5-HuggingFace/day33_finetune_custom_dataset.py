# ============================================
# DAY 33 - Fine Tune Small LLM on Custom Dataset
# Using GPT-2 + LoRA + Custom QA Dataset
# Author: Prateek Kumar Kuntal
# Date: 06 June 2026
# ============================================

import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)
from datasets import Dataset
from trl import SFTTrainer
import json
import os


# ------------------------------------------
# PART 1 - WHAT IS INSTRUCTION FINE TUNING
# ------------------------------------------

print("===== PART 1: What is Instruction Fine Tuning =====")

print("""
INSTRUCTION FINE TUNING:
    Train LLM to follow human instructions
    Transform base model into assistant model
    Base GPT-2 just predicts next token
    After instruction fine tuning it answers questions

    This is exactly how ChatGPT was made from GPT
    GPT-3 base model -> instruction fine tuned -> ChatGPT

DATA FORMAT:
    Instruction: the task description
    Input: optional context or input
    Output: the expected response

    Example:
    Instruction: "Classify the sentiment of this review"
    Input: "The food was absolutely amazing"
    Output: "Positive"

PROMPT TEMPLATE:
    Wrap data in consistent template
    Model learns to follow template structure
    At inference just provide instruction and input
    Model generates the output

ALPACA FORMAT (most popular):
    ### Instruction:
    {instruction}

    ### Input:
    {input}

    ### Response:
    {output}

WHY WE USE GPT-2 TODAY:
    Small model runs on CPU
    Same concepts apply to LLaMA, Mistral etc
    On Google Colab GPU you would use LLaMA 7B
    For now GPT-2 teaches the workflow perfectly
""")


# ------------------------------------------
# PART 2 - CREATE CUSTOM DATASET
# ------------------------------------------

print("===== PART 2: Create Custom Dataset =====")

# Custom AI/ML QA dataset
qa_data = [
    {
        "instruction": "What is machine learning?",
        "input": "",
        "output": "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. It uses algorithms to identify patterns and make predictions or decisions."
    },
    {
        "instruction": "Explain what a neural network is.",
        "input": "",
        "output": "A neural network is a computational model inspired by the human brain. It consists of layers of interconnected nodes called neurons that process information and learn to recognize patterns in data."
    },
    {
        "instruction": "What is the difference between supervised and unsupervised learning?",
        "input": "",
        "output": "Supervised learning uses labeled data where the correct answers are provided during training. Unsupervised learning finds hidden patterns in data without labeled examples. Supervised learning predicts outcomes while unsupervised learning discovers structure."
    },
    {
        "instruction": "What is overfitting in machine learning?",
        "input": "",
        "output": "Overfitting occurs when a model learns the training data too well including noise and irrelevant patterns. It performs well on training data but poorly on new unseen data. Solutions include regularization, dropout, and getting more training data."
    },
    {
        "instruction": "Explain gradient descent.",
        "input": "",
        "output": "Gradient descent is an optimization algorithm that minimizes a loss function by iteratively moving in the direction of steepest descent. It calculates gradients and updates model weights to reduce the prediction error step by step."
    },
    {
        "instruction": "What is a transformer model?",
        "input": "",
        "output": "A transformer is a deep learning architecture introduced in 2017 that uses attention mechanisms to process sequences in parallel. It replaced recurrent networks and became the foundation for BERT, GPT, and all modern language models."
    },
    {
        "instruction": "What is the attention mechanism?",
        "input": "",
        "output": "The attention mechanism allows models to focus on relevant parts of the input when producing each output. It computes weighted combinations of values using query and key similarity scores, enabling models to capture long range dependencies efficiently."
    },
    {
        "instruction": "What is BERT?",
        "input": "",
        "output": "BERT stands for Bidirectional Encoder Representations from Transformers. It is a transformer encoder model pretrained on masked language modeling. BERT reads entire sequences bidirectionally and excels at understanding tasks like classification and question answering."
    },
    {
        "instruction": "What is GPT?",
        "input": "",
        "output": "GPT stands for Generative Pretrained Transformer. It is a decoder only transformer trained on next token prediction. GPT generates text autoregressively and powers ChatGPT. Larger versions like GPT-4 have hundreds of billions of parameters."
    },
    {
        "instruction": "What is transfer learning?",
        "input": "",
        "output": "Transfer learning reuses knowledge from a model trained on one task for a different related task. Instead of training from scratch you start with pretrained weights and fine tune on your specific dataset. This saves enormous time and compute."
    },
    {
        "instruction": "Explain what LoRA is.",
        "input": "",
        "output": "LoRA stands for Low Rank Adaptation. It is a parameter efficient fine tuning method that adds small trainable matrices to frozen pretrained model layers. LoRA reduces trainable parameters by thousands of times while achieving similar performance to full fine tuning."
    },
    {
        "instruction": "What is a large language model?",
        "input": "",
        "output": "A large language model is a neural network trained on massive amounts of text data with billions of parameters. LLMs like GPT-4 and Claude learn to understand and generate human language and can perform tasks across many domains without task specific training."
    },
    {
        "instruction": "What is RAG?",
        "input": "",
        "output": "RAG stands for Retrieval Augmented Generation. It combines information retrieval with language model generation. The model retrieves relevant documents from a knowledge base and uses them as context to generate more accurate and grounded responses."
    },
    {
        "instruction": "What is a vector database?",
        "input": "",
        "output": "A vector database stores data as high dimensional vectors called embeddings. It enables semantic similarity search by finding vectors closest to a query vector. Examples include FAISS, Pinecone, and ChromaDB. They are essential for RAG applications."
    },
    {
        "instruction": "What is prompt engineering?",
        "input": "",
        "output": "Prompt engineering is the practice of designing effective inputs for language models to get desired outputs. Techniques include zero shot, few shot, chain of thought, and role prompting. Good prompts significantly improve LLM performance without any fine tuning."
    },
    {
        "instruction": "Classify the sentiment of this text.",
        "input": "The product quality is outstanding and delivery was super fast.",
        "output": "Positive. The text expresses satisfaction with both the product quality and delivery speed."
    },
    {
        "instruction": "Classify the sentiment of this text.",
        "input": "Terrible experience. The item broke after one day of use.",
        "output": "Negative. The text expresses strong dissatisfaction with poor product durability."
    },
    {
        "instruction": "Summarize the following text in one sentence.",
        "input": "Machine learning is transforming industries by enabling computers to learn patterns from data automatically. Companies are using it for image recognition, natural language processing, and predictive analytics to improve their products and services.",
        "output": "Machine learning is revolutionizing industries by allowing computers to automatically learn from data for applications like image recognition and predictive analytics."
    },
    {
        "instruction": "What programming language is best for machine learning?",
        "input": "",
        "output": "Python is the best programming language for machine learning. It has an extensive ecosystem of libraries like PyTorch, TensorFlow, scikit-learn, and HuggingFace Transformers. Its simple syntax allows data scientists to focus on algorithms rather than programming details."
    },
    {
        "instruction": "Explain what HuggingFace is.",
        "input": "",
        "output": "HuggingFace is an AI company and open source platform that provides tools for building machine learning applications. Their Transformers library has over 500000 pretrained models for NLP, vision, and audio tasks. It is the most popular platform for sharing and using AI models."
    },
]

print(f"Custom QA dataset created:")
print(f"Total examples   : {len(qa_data)}")
print(f"\nSample example:")
print(f"Instruction: {qa_data[0]['instruction']}")
print(f"Input      : {qa_data[0]['input'] or 'None'}")
print(f"Output     : {qa_data[0]['output'][:80]}...")


# ------------------------------------------
# PART 3 - FORMAT DATASET WITH PROMPT TEMPLATE
# ------------------------------------------

print("\n===== PART 3: Format Dataset with Prompt Template =====")

PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

PROMPT_TEMPLATE_NO_INPUT = """### Instruction:
{instruction}

### Response:
{output}"""

def format_prompt(example):
    if example["input"] and example["input"].strip():
        return PROMPT_TEMPLATE.format(
            instruction = example["instruction"],
            input       = example["input"],
            output      = example["output"]
        )
    else:
        return PROMPT_TEMPLATE_NO_INPUT.format(
            instruction = example["instruction"],
            output      = example["output"]
        )

# Format all examples
formatted_texts = [format_prompt(ex) for ex in qa_data]

print("Formatted prompt example:")
print("=" * 50)
print(formatted_texts[0])
print("=" * 50)

print(f"\nFormatted prompt example with input:")
print("=" * 50)
print(formatted_texts[15])
print("=" * 50)

# Check lengths
lengths = [len(text.split()) for text in formatted_texts]
print(f"\nDataset statistics:")
print(f"Min length       : {min(lengths)} words")
print(f"Max length       : {max(lengths)} words")
print(f"Avg length       : {np.mean(lengths):.1f} words")


# ------------------------------------------
# PART 4 - LOAD MODEL AND TOKENIZER
# ------------------------------------------

print("\n===== PART 4: Load Model and Tokenizer =====")

model_name = "gpt2"
print(f"Loading {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype = torch.float32,
)
model.config.pad_token_id = tokenizer.eos_token_id

total_params = sum(p.numel() for p in model.parameters())
print(f"Model loaded!")
print(f"Parameters       : {total_params:,}")
print(f"Model dtype      : {next(model.parameters()).dtype}")


# ------------------------------------------
# PART 5 - APPLY LORA TO GPT-2
# ------------------------------------------

print("\n===== PART 5: Apply LoRA to GPT-2 =====")

lora_config = LoraConfig(
    task_type     = TaskType.CAUSAL_LM,
    r             = 8,
    lora_alpha    = 16,
    lora_dropout  = 0.05,
    bias          = "none",
    target_modules= ["c_attn", "c_proj"],  # GPT-2 attention
)

model = get_peft_model(model, lora_config)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)

print(f"LoRA applied!")
print(f"Total parameters    : {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Trainable %         : "
      f"{trainable_params/total_params*100:.3f}%")

model.print_trainable_parameters()


# ------------------------------------------
# PART 6 - CREATE HUGGINGFACE DATASET
# ------------------------------------------

print("\n===== PART 6: Create HuggingFace Dataset =====")

# Split into train and validation
split_idx    = int(0.85 * len(formatted_texts))
train_texts  = formatted_texts[:split_idx]
val_texts    = formatted_texts[split_idx:]

train_dataset = Dataset.from_dict({"text": train_texts})
val_dataset   = Dataset.from_dict({"text": val_texts})

print(f"Train dataset    : {len(train_dataset)} examples")
print(f"Val dataset      : {len(val_dataset)} examples")

# Tokenize datasets
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation  = True,
        max_length  = 256,
        padding     = "max_length",
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

train_tokenized = train_dataset.map(
    tokenize_function, batched=True,
    remove_columns=["text"])
val_tokenized   = val_dataset.map(
    tokenize_function, batched=True,
    remove_columns=["text"])

train_tokenized.set_format("torch")
val_tokenized.set_format("torch")

print(f"\nTokenized train  : {train_tokenized}")
print(f"Columns          : {train_tokenized.column_names}")


# ------------------------------------------
# PART 7 - TRAINING
# ------------------------------------------

print("\n===== PART 7: Training =====")

training_args = TrainingArguments(
    output_dir                  = "gpt2_lora_aiml_tutor",
    num_train_epochs            = 10,
    per_device_train_batch_size = 2,
    per_device_eval_batch_size  = 2,
    gradient_accumulation_steps = 4,
    learning_rate               = 3e-4,
    weight_decay                = 0.01,
    warmup_ratio                = 0.1,
    lr_scheduler_type           = "cosine",
    evaluation_strategy         = "epoch",
    save_strategy               = "epoch",
    load_best_model_at_end      = True,
    metric_for_best_model       = "eval_loss",
    greater_is_better           = False,
    logging_steps               = 5,
    report_to                   = "none",
    seed                        = 42,
    fp16                        = False,   # CPU training
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer = tokenizer,
    mlm       = False,     # causal LM not masked LM
)

from transformers import Trainer as HFTrainer

trainer = HFTrainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_tokenized,
    eval_dataset    = val_tokenized,
    data_collator   = data_collator,
)

print(f"Training configuration:")
print(f"  Epochs           : {training_args.num_train_epochs}")
print(f"  Batch size       : {training_args.per_device_train_batch_size}")
print(f"  Grad accumulation: {training_args.gradient_accumulation_steps}")
print(f"  Learning rate    : {training_args.learning_rate}")
print(f"  LR Scheduler     : {training_args.lr_scheduler_type}")
print(f"\nStarting training...")
print(f"{'='*60}")

train_result = trainer.train()

print(f"\nTraining complete!")
print(f"Training loss    : {train_result.training_loss:.4f}")
print(f"Training time    : "
      f"{train_result.metrics['train_runtime']:.1f}s")


# ------------------------------------------
# PART 8 - GENERATE RESPONSES
# ------------------------------------------

print("\n===== PART 8: Generate Responses =====")

def generate_response(instruction, input_text="",
                       model=model, tokenizer=tokenizer,
                       max_new_tokens=150):
    model.eval()

    if input_text.strip():
        prompt = PROMPT_TEMPLATE.format(
            instruction = instruction,
            input       = input_text,
            output      = ""
        )
    else:
        prompt = PROMPT_TEMPLATE_NO_INPUT.format(
            instruction = instruction,
            output      = ""
        )

    # Remove trailing empty output marker
    prompt = prompt.rstrip()

    inputs  = tokenizer.encode(
        prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens     = max_new_tokens,
            do_sample          = True,
            temperature        = 0.7,
            top_p              = 0.9,
            repetition_penalty = 1.3,
            pad_token_id       = tokenizer.eos_token_id,
        )

    full_response = tokenizer.decode(
        outputs[0], skip_special_tokens=True)

    # Extract only the response part
    if "### Response:" in full_response:
        response = full_response.split("### Response:")[-1].strip()
    else:
        response = full_response[len(prompt):].strip()

    return response

# Test the fine tuned model
test_questions = [
    ("What is machine learning?", ""),
    ("What is the transformer architecture?", ""),
    ("Explain LoRA fine tuning.", ""),
    ("What is RAG?", ""),
    ("Classify the sentiment of this text.",
     "The product is absolutely amazing and works perfectly."),
]

print("Fine tuned model responses:")
print("=" * 70)

for instruction, input_text in test_questions:
    print(f"\nInstruction: {instruction}")
    if input_text:
        print(f"Input      : {input_text}")

    response = generate_response(
        instruction, input_text)
    print(f"Response   : {response[:200]}")
    print("-" * 70)


# ------------------------------------------
# PART 9 - COMPARE BASE VS FINE TUNED
# ------------------------------------------

print("\n===== PART 9: Compare Base vs Fine Tuned =====")

# Load base GPT-2 for comparison
print("Loading base GPT-2 for comparison...")
base_tokenizer = AutoTokenizer.from_pretrained("gpt2")
base_model     = AutoModelForCausalLM.from_pretrained("gpt2")
base_tokenizer.pad_token = base_tokenizer.eos_token
base_model.eval()

def generate_base(prompt, max_new_tokens=80):
    inputs = base_tokenizer.encode(
        prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = base_model.generate(
            inputs,
            max_new_tokens     = max_new_tokens,
            do_sample          = True,
            temperature        = 0.7,
            top_p              = 0.9,
            repetition_penalty = 1.2,
            pad_token_id       = base_tokenizer.eos_token_id,
        )

    return base_tokenizer.decode(
        outputs[0], skip_special_tokens=True)

test_prompt = "What is machine learning?"

print(f"\nTest question: {test_prompt}")
print(f"\nBase GPT-2 (no fine tuning):")
base_response = generate_base(test_prompt)
print(f"  {base_response}")

print(f"\nFine tuned GPT-2 (instruction tuned):")
ft_response = generate_response(test_prompt)
print(f"  {ft_response[:300]}")

print(f"""
Observation:
    Base GPT-2 generates text that continues the question
    Fine tuned GPT-2 actually attempts to answer it
    With a larger model like LLaMA 7B on Colab GPU
    the fine tuned model would give excellent answers
""")


# ------------------------------------------
# PART 10 - SAVE FINE TUNED MODEL
# ------------------------------------------

print("===== PART 10: Save Fine Tuned Model =====")

save_path = "gpt2_aiml_tutor_lora"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

files = os.listdir(save_path)
print(f"Model saved to {save_path}/")
print(f"Files saved:")
for f in files:
    size = os.path.getsize(
        os.path.join(save_path, f))
    print(f"  {f:<40} {size/1024:.1f} KB")

total_size = sum(
    os.path.getsize(os.path.join(save_path, f))
    for f in files
) / 1024 / 1024

print(f"\nTotal adapter size   : {total_size:.2f} MB")
print(f"(Full GPT-2 is 548 MB, adapter is tiny!)")


# ------------------------------------------
# MINI PROJECT - AIML Tutor Chatbot
# ------------------------------------------

print("\n===== MINI PROJECT: AIML Tutor Chatbot =====")

class AIMLTutorChatbot:
    def __init__(self, model, tokenizer):
        self.model       = model
        self.tokenizer   = tokenizer
        self.history     = []
        self.model.eval()

    def answer(self, question):
        response = generate_response(
            instruction    = question,
            input_text     = "",
            model          = self.model,
            tokenizer      = self.tokenizer,
            max_new_tokens = 120
        )
        self.history.append({
            "question": question,
            "answer"  : response
        })
        return response

    def get_history(self):
        return self.history

    def evaluate_knowledge(self, questions):
        print("Evaluating chatbot knowledge...")
        print("=" * 60)
        for q in questions:
            answer = self.answer(q)
            print(f"\nQ: {q}")
            print(f"A: {answer[:200]}")
            print("-" * 60)

chatbot = AIMLTutorChatbot(model, tokenizer)

eval_questions = [
    "What is deep learning?",
    "How does the attention mechanism work?",
    "What is the difference between BERT and GPT?",
    "What is fine tuning in machine learning?",
    "Explain what HuggingFace is and why it is useful.",
]

chatbot.evaluate_knowledge(eval_questions)

print(f"\nChatbot conversation history:")
print(f"Total questions answered: {len(chatbot.get_history())}")
print(f"\nKey takeaway:")
print(f"This same workflow works for LLaMA 7B on Colab GPU")
print(f"Replace gpt2 with meta-llama/Llama-2-7b-hf")
print(f"Add QLoRA quantization config")
print(f"Result: your own custom LLM tutor!")


print("\n===== WHAT I LEARNED TODAY =====")
print("Instruction fine tuning concept and importance")
print("Alpaca prompt template format")
print("Creating custom QA dataset for fine tuning")
print("Applying LoRA to GPT-2 for causal LM task")
print("DataCollatorForLanguageModeling for causal LM")
print("Training with gradient accumulation")
print("Generating responses with fine tuned model")
print("Comparing base vs fine tuned model output")
print("Saving tiny LoRA adapter vs full model")
print("Mini Project - AIML Tutor Chatbot")
print("\nDay 33 Done! Tomorrow - Prompt Engineering!")