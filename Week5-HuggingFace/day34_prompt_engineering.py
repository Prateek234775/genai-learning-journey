# ============================================
# DAY 34 - Prompt Engineering
# Zero Shot, Few Shot, Chain of Thought
# Author: Prateek Kumar Kuntal
# Date: 07 June 2026
# ============================================

import os
import json


# ------------------------------------------
# PART 1 - WHAT IS PROMPT ENGINEERING
# ------------------------------------------

print("===== PART 1: What is Prompt Engineering =====")

print("""
PROMPT ENGINEERING:
    The art of designing effective inputs for LLMs
    Same model gives completely different outputs
    based on how you phrase the prompt
    No training required - just better prompts

    Bad prompt  -> bad output
    Good prompt -> great output

WHY IT MATTERS:
    LLMs are extremely sensitive to wording
    A well crafted prompt can unlock capabilities
    that seem impossible with a bad prompt
    Saves enormous time and cost vs fine tuning

WHO USES PROMPT ENGINEERING:
    Every company using LLMs in production
    ChatGPT system prompt is prompt engineering
    GitHub Copilot uses carefully crafted prompts
    Every AI product has a prompt engineer

PROMPT ANATOMY:
    System message  - sets the AI persona and rules
    Context         - background information
    Instruction     - what you want the AI to do
    Input           - the data to process
    Output format   - how you want the response
    Examples        - demonstrate desired behavior
""")


# ------------------------------------------
# PART 2 - ZERO SHOT PROMPTING
# ------------------------------------------

print("===== PART 2: Zero Shot Prompting =====")

print("""
ZERO SHOT PROMPTING:
    Give the model a task with no examples
    Relies entirely on pretrained knowledge
    Works well for common tasks
    Simplest form of prompting

    Just describe what you want clearly
    and the model does its best
""")

# We demonstrate prompt engineering concepts
# using structured examples since we do not have
# API access. These show exactly what to send to
# OpenAI API, Anthropic API or any LLM API.

zero_shot_prompts = {
    "Sentiment Classification": {
        "prompt": """Classify the sentiment of the following text.
Answer with only one word: Positive, Negative, or Neutral.

Text: The new iPhone camera is absolutely incredible. Photos look stunning.
Sentiment:""",
        "expected_output": "Positive"
    },

    "Language Detection": {
        "prompt": """Detect the language of the following text.
Answer with only the language name.

Text: Bonjour, comment allez-vous aujourd'hui?
Language:""",
        "expected_output": "French"
    },

    "Topic Classification": {
        "prompt": """Classify the topic of this news headline.
Choose from: Technology, Sports, Politics, Business, Entertainment.

Headline: India wins cricket world cup defeating Australia in final
Topic:""",
        "expected_output": "Sports"
    },

    "Named Entity": {
        "prompt": """Extract all person names from the following text.
List them separated by commas.

Text: Prateek Kumar met with Elon Musk and Sundar Pichai at the AI summit in San Francisco.
Person names:""",
        "expected_output": "Prateek Kumar, Elon Musk, Sundar Pichai"
    },
}

print("Zero Shot Prompt Examples:")
print("=" * 70)
for task, data in zero_shot_prompts.items():
    print(f"\nTask: {task}")
    print(f"Prompt:\n{data['prompt']}")
    print(f"Expected output: {data['expected_output']}")
    print("-" * 70)


# ------------------------------------------
# PART 3 - FEW SHOT PROMPTING
# ------------------------------------------

print("\n===== PART 3: Few Shot Prompting =====")

print("""
FEW SHOT PROMPTING:
    Provide a few examples before the actual task
    Examples demonstrate the desired input-output format
    Model learns the pattern from examples
    Much more reliable than zero shot for complex tasks

    Show dont tell principle
    3 to 5 examples is usually enough
    Examples should be diverse and representative
""")

few_shot_prompts = {
    "Sentiment with Examples": """Classify sentiment as Positive, Negative, or Neutral.

Text: The food was absolutely delicious
Sentiment: Positive

Text: Service was okay nothing special
Sentiment: Neutral

Text: Worst experience ever complete waste of money
Sentiment: Negative

Text: The hotel room was clean and staff were very helpful
Sentiment:""",

    "Code Bug Detection": """Identify if the following Python code has a bug.
Answer with Yes or No and briefly explain.

Code: def add(a, b): return a + b
Bug: No - This function correctly adds two numbers.

Code: def divide(a, b): return a / b
Bug: Yes - No check for division by zero when b is 0.

Code: for i in range(10): print(i
Bug: Yes - Missing closing parenthesis on print statement.

Code: def greet(name): return "Hello " + name
Bug:""",

    "SQL Query Generation": """Convert the natural language request to a SQL query.

Request: Get all users older than 25
SQL: SELECT * FROM users WHERE age > 25;

Request: Count total orders per customer
SQL: SELECT customer_id, COUNT(*) as total_orders FROM orders GROUP BY customer_id;

Request: Find top 5 products by sales
SQL: SELECT product_name, SUM(quantity) as total_sales FROM orders GROUP BY product_name ORDER BY total_sales DESC LIMIT 5;

Request: Get all employees in the Engineering department
SQL:""",

    "Email Subject Generation": """Generate a professional email subject line.

Email body: I wanted to follow up on our meeting last week regarding the Q3 budget proposal and get your thoughts.
Subject: Follow-up: Q3 Budget Proposal Discussion

Email body: We are excited to announce that our new product will launch next month and wanted to share the details with you.
Subject: Exciting News: New Product Launch Next Month

Email body: I noticed an error in the invoice you sent and would like to discuss the correct amount before payment.
Subject: Invoice Discrepancy - Action Required

Email body: Thank you for attending our webinar yesterday. Here are the resources and recording we promised.
Subject:""",
}

print("Few Shot Prompt Examples:")
print("=" * 70)
for task, prompt in few_shot_prompts.items():
    print(f"\nTask: {task}")
    print(f"Prompt:\n{prompt}")
    print("-" * 70)


# ------------------------------------------
# PART 4 - CHAIN OF THOUGHT PROMPTING
# ------------------------------------------

print("\n===== PART 4: Chain of Thought Prompting =====")

print("""
CHAIN OF THOUGHT (CoT):
    Ask the model to think step by step
    Show reasoning process before final answer
    Dramatically improves complex reasoning tasks
    Especially useful for math, logic, multi step problems

    "Let's think step by step" is a magical phrase
    Adding this to your prompt often improves accuracy

TWO TYPES:
    Zero shot CoT - just say "think step by step"
    Few shot CoT  - show examples with reasoning steps

WHEN TO USE:
    Math word problems
    Logic puzzles
    Multi step reasoning
    Complex decision making
    Any task that requires intermediate reasoning
""")

cot_prompts = {
    "Math Problem - Zero Shot CoT": """Solve this math problem. Think step by step.

Problem: A train travels 120 km in 2 hours then 180 km in 3 hours.
What is the average speed for the entire journey?

Solution:""",

    "Math Problem - Few Shot CoT": """Solve math problems step by step.

Problem: A store sells apples for 5 rupees each. If I buy 8 apples and pay with 50 rupees, how much change do I get?
Solution: 
Step 1: Calculate total cost. 8 apples x 5 rupees = 40 rupees
Step 2: Calculate change. 50 - 40 = 10 rupees
Answer: 10 rupees change

Problem: A class has 30 students. 40 percent passed the exam. How many students failed?
Solution:
Step 1: Calculate students who passed. 30 x 0.40 = 12 students passed
Step 2: Calculate students who failed. 30 - 12 = 18 students failed
Answer: 18 students failed

Problem: Prateek studies 3 hours on weekdays and 5 hours on weekends. How many hours does he study in 4 weeks?
Solution:""",

    "Logic Reasoning": """Answer logic questions by reasoning step by step.

Question: All roses are flowers. All flowers need water. Does a rose need water?
Reasoning:
Step 1: Roses are flowers (given)
Step 2: All flowers need water (given)
Step 3: Since roses are flowers and flowers need water, roses need water
Answer: Yes, a rose needs water

Question: If it rains the ground gets wet. The ground is wet. Did it definitely rain?
Reasoning:""",

    "Code Debugging CoT": """Debug code by thinking through each step.

Problem: Find why this function returns wrong output.
Code: 
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total = total + num
    return total / 10

Input: [10, 20, 30]
Expected: 20.0
Actual: 6.0

Debugging:
Step 1: Check the loop - it correctly adds all numbers. total = 60
Step 2: Check the division - divides by 10 always instead of len(numbers)
Step 3: The bug is dividing by hardcoded 10 instead of len(numbers)
Fix: return total / len(numbers)

Problem: Find why this function fails.
Code:
def find_max(lst):
    max_val = 0
    for item in lst:
        if item > max_val:
            max_val = item
    return max_val

Input: [-5, -2, -8, -1]
Expected: -1
Actual: 0

Debugging:""",
}

print("Chain of Thought Prompt Examples:")
print("=" * 70)
for task, prompt in cot_prompts.items():
    print(f"\nTask: {task}")
    print(f"Prompt:\n{prompt}")
    print("-" * 70)


# ------------------------------------------
# PART 5 - SYSTEM PROMPTS
# ------------------------------------------

print("\n===== PART 5: System Prompts =====")

print("""
SYSTEM PROMPT:
    Sets the persona, rules and behavior of the AI
    Applied before every user message
    The most powerful form of prompt engineering
    This is how ChatGPT becomes a specific assistant

WHAT TO PUT IN SYSTEM PROMPT:
    Role and persona definition
    Tone and communication style
    Rules and constraints
    Knowledge domain
    Output format requirements
    Things to always or never do

EXAMPLES OF SYSTEM PROMPTS:
    ChatGPT   - "You are a helpful assistant..."
    GitHub Copilot - "You are a coding assistant..."
    Customer service bot - "You are a support agent for..."
""")

system_prompts = {
    "AIML Tutor": """You are an expert AI and Machine Learning tutor named Alex.
You teach students who are learning AI/ML from scratch.

Rules:
- Always explain concepts in simple terms first then go deeper
- Use real world analogies to explain complex ideas
- Give Python code examples when relevant
- Encourage students and be patient
- If unsure about something say so honestly
- Keep responses concise but complete

Your student is a 2nd year B.Tech student learning GenAI engineering.""",

    "Code Reviewer": """You are a senior software engineer conducting code reviews.

Your responsibilities:
- Identify bugs and logical errors
- Suggest performance improvements
- Check for security vulnerabilities
- Ensure code follows best practices
- Provide specific actionable feedback

Output format:
- Start with overall assessment
- List issues found with line numbers
- Suggest specific improvements
- End with positive observations

Be direct and constructive. Do not sugarcoat issues.""",

    "Data Analyst": """You are a senior data analyst with expertise in Python and statistics.

When analyzing data:
- Always start with data overview and quality check
- Identify key patterns and anomalies
- Use statistical reasoning to support findings
- Create clear visualizations when relevant
- Provide actionable business insights

Output format: structured report with sections
Tone: professional and data driven
Always show your analytical reasoning.""",

    "Customer Support": """You are a friendly customer support agent for TechStore India.

Products: laptops, phones, tablets, accessories
Policies:
- Returns accepted within 30 days with receipt
- Warranty: 1 year for all products
- Delivery: 3-5 business days
- Payment: all major cards, UPI, net banking

Rules:
- Always greet customer warmly
- Show empathy for their problems
- Never make promises you cannot keep
- Escalate complex issues to supervisor
- End every conversation asking if there is anything else

Language: formal but friendly Hindi-English mix is acceptable""",
}

print("System Prompt Examples:")
print("=" * 70)
for role, prompt in system_prompts.items():
    print(f"\nRole: {role}")
    print(f"System Prompt:\n{prompt}")
    print("-" * 70)


# ------------------------------------------
# PART 6 - ADVANCED TECHNIQUES
# ------------------------------------------

print("\n===== PART 6: Advanced Prompting Techniques =====")

print("""
ROLE PROMPTING:
    Assign a specific expert role to the model
    "You are an expert Python developer with 10 years experience"
    "You are a medical doctor specializing in cardiology"
    Model adopts the knowledge and style of that role

STRUCTURED OUTPUT:
    Request specific output format
    JSON, markdown, numbered lists, tables
    Makes outputs easier to parse programmatically
    Critical for production applications

SELF CONSISTENCY:
    Generate multiple answers independently
    Take majority vote as final answer
    Improves accuracy on reasoning tasks
    More expensive but more reliable

REACT PROMPTING:
    Reasoning and Acting combined
    Model reasons then takes an action
    Observes result then reasons again
    Foundation of AI agents like LangChain Agents

PROMPT CHAINING:
    Break complex task into smaller prompts
    Output of one prompt feeds into next
    More reliable than one giant prompt
    Easier to debug individual steps

DELIMITERS:
    Use clear separators for different parts
    Triple quotes, XML tags, dashes
    Helps model understand input structure
    Reduces prompt injection attacks
""")

advanced_examples = {
    "Role Prompting": """You are a senior Python developer with 10 years of experience
who specializes in writing clean maintainable code.

Review this code and suggest improvements as an expert would:

def calc(x,y,z):
    r = x+y
    if r > z:
        return True
    else:
        return False""",

    "Structured JSON Output": """Extract information from the following job posting and return it as JSON.

Job Posting:
We are hiring a Senior Machine Learning Engineer at TechCorp Bangalore.
Required experience: 5 years. Salary: 25-40 LPA.
Skills needed: Python, PyTorch, MLOps, Docker.
Application deadline: June 30 2025.

Return ONLY valid JSON with no explanation in this format:
{
    "title": "",
    "company": "",
    "location": "",
    "experience_years": 0,
    "salary_range": "",
    "skills": [],
    "deadline": ""
}""",

    "Prompt Chaining Step 1": """Step 1 of 3: Extract the main topic from this customer complaint.
Return only the topic in 3 words or less.

Complaint: I ordered a laptop last week but received a tablet instead.
I have tried calling customer service three times but no one answers.
I need this resolved urgently as I need the laptop for work.

Main topic:""",

    "Prompt Chaining Step 2": """Step 2 of 3: Given this customer complaint topic, generate the appropriate
department to handle it and the priority level.

Topic: Wrong item delivered
Return in format: Department: [name], Priority: [High/Medium/Low]

Department and Priority:""",

    "Prompt Chaining Step 3": """Step 3 of 3: Write a professional customer response email.

Customer complaint: Ordered laptop received tablet, cannot reach support.
Department handling: Order Fulfillment
Priority: High
Company: TechStore India

Email:""",

    "Delimiter Usage": """Analyze the sentiment of ONLY the customer review between the triple dashes.
Ignore any other text.

Background context (do not analyze this):
This review is from our e-commerce platform for internal training purposes.

---
The delivery was super fast and the packaging was perfect. The phone works
exactly as described and the camera quality is outstanding. Will definitely
order again from this store.
---

Sentiment of the review:""",
}

print("Advanced Prompting Examples:")
print("=" * 70)
for technique, prompt in advanced_examples.items():
    print(f"\nTechnique: {technique}")
    print(f"Prompt:\n{prompt}")
    print("-" * 70)


# ------------------------------------------
# PART 7 - PROMPT ANTI-PATTERNS
# ------------------------------------------

print("\n===== PART 7: Prompt Anti-Patterns to Avoid =====")

anti_patterns = {
    "Vague Instructions": {
        "bad" : "Write something about AI.",
        "good": "Write a 3 paragraph explanation of how transformer models work for a B.Tech student with basic Python knowledge. Use simple language and include one real world analogy.",
        "why" : "Specific instructions give specific outputs"
    },
    "No Format Specification": {
        "bad" : "List the advantages of Python for ML.",
        "good": "List exactly 5 advantages of Python for machine learning. Format as a numbered list. Each point should be one sentence.",
        "why" : "Specifying format makes parsing and display easier"
    },
    "Ambiguous Pronouns": {
        "bad" : "Compare BERT and GPT. Explain how it works.",
        "good": "Compare BERT and GPT. Explain how BERT works, then explain how GPT works separately.",
        "why" : "Unclear pronouns confuse the model"
    },
    "Too Many Tasks at Once": {
        "bad" : "Analyze this code, fix bugs, add documentation, optimize performance and convert to Python 3.",
        "good": "First analyze this code and list all bugs found. I will ask for fixes in the next message.",
        "why" : "Break complex tasks into smaller focused prompts"
    },
    "Negative Instructions Only": {
        "bad" : "Do not use technical jargon. Do not be verbose. Do not give examples.",
        "good": "Explain in simple everyday language. Be concise in 2-3 sentences. Focus only on the core concept.",
        "why" : "Tell model what TO do not just what NOT to do"
    },
}

print("Prompt Anti-Patterns:")
print("=" * 70)
for pattern, data in anti_patterns.items():
    print(f"\nAnti-Pattern: {pattern}")
    print(f"  Bad  : {data['bad']}")
    print(f"  Good : {data['good']}")
    print(f"  Why  : {data['why']}")


# ------------------------------------------
# PART 8 - PROMPT TEMPLATE LIBRARY
# ------------------------------------------

print("\n===== PART 8: Prompt Template Library =====")

class PromptLibrary:
    def __init__(self):
        self.templates = {}

    def add_template(self, name, template, description):
        self.templates[name] = {
            "template"   : template,
            "description": description,
        }

    def get_prompt(self, name, **kwargs):
        if name not in self.templates:
            raise ValueError(f"Template {name} not found")
        return self.templates[name]["template"].format(**kwargs)

    def list_templates(self):
        print("Available templates:")
        for name, data in self.templates.items():
            print(f"  {name:<30} : {data['description']}")

    def save_to_file(self, filepath):
        save_data = {
            name: {
                "template"   : data["template"],
                "description": data["description"],
            }
            for name, data in self.templates.items()
        }
        with open(filepath, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"Templates saved to {filepath}")

    def load_from_file(self, filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
        for name, template_data in data.items():
            self.templates[name] = template_data
        print(f"Templates loaded from {filepath}")


# Build a prompt library
library = PromptLibrary()

library.add_template(
    name        = "sentiment_analysis",
    template    = """Classify the sentiment of the following text.
Return only one word: Positive, Negative, or Neutral.

Text: {text}
Sentiment:""",
    description = "Simple sentiment classification"
)

library.add_template(
    name        = "code_review",
    template    = """You are a senior {language} developer.
Review the following code for bugs, performance issues and best practices.
Provide specific actionable feedback.

Code:
{code}

Review:""",
    description = "Code review with language specification"
)

library.add_template(
    name        = "summarize",
    template    = """Summarize the following text in {num_sentences} sentences.
Focus on the most important points.
Write at a {level} reading level.

Text:
{text}

Summary:""",
    description = "Configurable text summarization"
)

library.add_template(
    name        = "question_generation",
    template    = """Generate {num_questions} multiple choice questions about the following topic.
Each question should have 4 options labeled A B C D.
Indicate the correct answer at the end.

Topic: {topic}
Difficulty: {difficulty}

Questions:""",
    description = "MCQ generation for any topic"
)

library.add_template(
    name        = "explain_concept",
    template    = """Explain {concept} to a {audience}.
Use simple language and a real world analogy.
Keep the explanation under {max_words} words.

Explanation:""",
    description = "Concept explanation for different audiences"
)

library.add_template(
    name        = "data_extraction",
    template    = """Extract the following information from the text and return as JSON.
Fields to extract: {fields}
If a field is not found return null.

Text:
{text}

JSON:""",
    description = "Structured data extraction to JSON"
)

# List all templates
library.list_templates()

# Use templates
print(f"\nUsing prompt templates:")
print(f"\n1. Sentiment Analysis:")
prompt = library.get_prompt(
    "sentiment_analysis",
    text="The customer service was absolutely terrible and very rude"
)
print(prompt)

print(f"\n2. Explain Concept:")
prompt = library.get_prompt(
    "explain_concept",
    concept      = "gradient descent",
    audience     = "10 year old child",
    max_words    = 100
)
print(prompt)

print(f"\n3. Question Generation:")
prompt = library.get_prompt(
    "question_generation",
    num_questions = 3,
    topic         = "Python decorators",
    difficulty    = "intermediate"
)
print(prompt)

# Save library to file
library.save_to_file("prompt_library.json")


# ------------------------------------------
# MINI PROJECT - Production Prompt System
# ------------------------------------------

print("\n===== MINI PROJECT: Production Prompt System =====")

class ProductionPromptSystem:
    def __init__(self, system_prompt, library):
        self.system_prompt = system_prompt
        self.library       = library
        self.call_log      = []

    def build_messages(self, user_message):
        return [
            {"role": "system",  "content": self.system_prompt},
            {"role": "user",    "content": user_message},
        ]

    def build_few_shot_messages(self, examples,
                                 user_message):
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        for ex in examples:
            messages.append(
                {"role": "user",      "content": ex["input"]})
            messages.append(
                {"role": "assistant", "content": ex["output"]})
        messages.append(
            {"role": "user", "content": user_message})
        return messages

    def log_call(self, prompt_name, user_input, response):
        self.call_log.append({
            "prompt_name": prompt_name,
            "user_input" : user_input,
            "response"   : response,
            "tokens_est" : len(user_input.split()) +
                           len(response.split())
        })

    def get_stats(self):
        total_calls  = len(self.call_log)
        total_tokens = sum(c["tokens_est"]
                           for c in self.call_log)
        return {
            "total_calls"  : total_calls,
            "total_tokens" : total_tokens,
            "avg_tokens"   : total_tokens / max(total_calls, 1)
        }

# Build production system
aiml_tutor_system = system_prompts["AIML Tutor"]
prod_system       = ProductionPromptSystem(
    system_prompt = aiml_tutor_system,
    library       = library
)

# Simulate production calls
test_inputs = [
    "What is machine learning?",
    "Explain neural networks simply",
    "What is the difference between AI and ML?",
    "How does GPT generate text?",
    "What should I learn first for GenAI?",
]

simulated_responses = [
    "Machine learning is teaching computers to learn from examples rather than following fixed rules.",
    "Neural networks are computing systems loosely inspired by the brain consisting of connected nodes.",
    "AI is the broad concept of machines being smart. ML is a specific way to achieve AI using data.",
    "GPT generates text one word at a time by predicting the most likely next word given all previous words.",
    "Start with Python basics then learn ML fundamentals then move to deep learning and transformers.",
]

print("Production Prompt System Demo:")
print("=" * 60)
for user_input, response in zip(
        test_inputs, simulated_responses):
    messages = prod_system.build_messages(user_input)
    prod_system.log_call(
        "aiml_tutor", user_input, response)

    print(f"\nUser    : {user_input}")
    print(f"System  : {aiml_tutor_system[:50]}...")
    print(f"Response: {response}")

stats = prod_system.get_stats()
print(f"\nSystem Statistics:")
print(f"  Total calls  : {stats['total_calls']}")
print(f"  Total tokens : {stats['total_tokens']}")
print(f"  Avg tokens   : {stats['avg_tokens']:.1f}")

# Few shot example
print(f"\nFew Shot Message Structure:")
few_shot_examples = [
    {"input" : "What is Python?",
     "output": "Python is a high level programming language known for its simple readable syntax."},
    {"input" : "What is PyTorch?",
     "output": "PyTorch is an open source deep learning framework developed by Meta used for building neural networks."},
]

messages = prod_system.build_few_shot_messages(
    few_shot_examples,
    "What is HuggingFace?"
)

print(f"\nMessage structure for API call:")
for msg in messages:
    role    = msg["role"].upper()
    content = msg["content"][:60]
    print(f"  [{role}]: {content}...")


print("\n===== WHAT I LEARNED TODAY =====")
print("What prompt engineering is and why it matters")
print("Zero shot prompting for simple tasks")
print("Few shot prompting for complex tasks")
print("Chain of thought for reasoning problems")
print("System prompts to define AI persona")
print("Role prompting for expert knowledge")
print("Structured JSON output from LLMs")
print("Prompt chaining for complex workflows")
print("Delimiter usage to structure prompts")
print("Common anti-patterns to avoid")
print("Building a reusable prompt template library")
print("Production prompt system with logging")
print("\nDay 34 Done! Tomorrow is REST DAY!")