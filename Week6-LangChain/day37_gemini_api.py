# ============================================
# DAY 37 - Gemini API Deep Dive
# Function Calling, Multimodal, Advanced Features
# Author: Prateek Kumar Kuntal
# Date: 10 June 2026
# ============================================

import os
import json
import requests
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found in .env file")

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)
print("Gemini API configured successfully!")


# ------------------------------------------
# PART 1 - GEMINI MODEL FAMILY
# ------------------------------------------

print("\n===== PART 1: Gemini Model Family =====")

print("""
GEMINI MODEL FAMILY:
    Google released Gemini in December 2023
    Most powerful AI model family from Google

MODELS AVAILABLE:
    gemini-1.5-flash   - fast and efficient (free tier)
                         best for most tasks
                         1M token context window

    gemini-1.5-pro     - most capable model
                         better reasoning and coding
                         2M token context window

    gemini-1.0-pro     - older model, still useful
                         good for basic tasks

    gemini-pro-vision  - multimodal understanding
                         images + text input

KEY FEATURES:
    Multimodal         - text, images, audio, video
    Long context       - up to 2M tokens (entire books!)
    Function calling   - call external functions
    Code execution     - run Python code
    Grounding         - connect to Google Search
    JSON mode         - structured output

FREE TIER LIMITS (gemini-1.5-flash):
    15 requests per minute
    1 million tokens per minute
    1500 requests per day
    More than enough for learning and projects
""")

# List available models
print("Available Gemini models:")
for model in genai.list_models():
    if "generateContent" in model.supported_generation_methods:
        print(f"  {model.name}")


# ------------------------------------------
# PART 2 - BASIC GEMINI API CALLS
# ------------------------------------------

print("\n===== PART 2: Basic Gemini API Calls =====")

# Direct Gemini API
model = genai.GenerativeModel("gemini-1.5-flash")

# Simple text generation
print("--- Simple Text Generation ---")
response = model.generate_content(
    "Explain what LangChain is in 3 sentences.")
print(f"Response: {response.text}")

# Generation config
print("\n--- Generation Config ---")
generation_config = genai.GenerationConfig(
    temperature      = 0.3,
    top_p            = 0.9,
    top_k            = 40,
    max_output_tokens= 200,
    candidate_count  = 1,
)

response = model.generate_content(
    "What are the top 5 skills for a GenAI engineer?",
    generation_config = generation_config
)
print(f"Response (temp=0.3):\n{response.text}")

# Safety settings
print("\n--- Token counting ---")
prompt    = "Explain transformer architecture in detail"
response  = model.generate_content(prompt)
print(f"Prompt           : {prompt}")
print(f"Response tokens  : {model.count_tokens(prompt)}")
print(f"Response preview : {response.text[:100]}...")


# ------------------------------------------
# PART 3 - CHAT WITH GEMINI
# ------------------------------------------

print("\n===== PART 3: Multi-turn Chat =====")

print("""
CHAT SESSION:
    Gemini maintains conversation history natively
    No need for LangChain memory in simple cases
    start_chat() creates a chat session
    chat.send_message() sends messages

    History stored in chat.history
    Can initialize with existing history
""")

# Create chat session
chat = model.start_chat(history=[])

conversations = [
    "Hi I am Prateek. I am a B.Tech student learning GenAI.",
    "What topics should I focus on to become a GenAI engineer?",
    "What is my name and what am I studying?",
    "Suggest a project I can build this week.",
]

print("Multi-turn chat with Gemini:")
print("=" * 60)
for message in conversations:
    print(f"\nPrateek: {message}")
    response = chat.send_message(message)
    print(f"Gemini : {response.text}")
    print("-" * 60)

print(f"\nChat history length: {len(chat.history)} messages")

# Show history structure
print("\nChat history structure:")
for i, msg in enumerate(chat.history[:4]):
    role    = msg.role
    content = msg.parts[0].text[:50]
    print(f"  [{i+1}] {role.upper()}: {content}...")


# ------------------------------------------
# PART 4 - SYSTEM INSTRUCTIONS
# ------------------------------------------

print("\n===== PART 4: System Instructions =====")

print("""
SYSTEM INSTRUCTIONS:
    Define model behavior before conversation
    Set persona, rules, and constraints
    Much more powerful than putting in first message

    model = genai.GenerativeModel(
        model_name         = "gemini-1.5-flash",
        system_instruction = "You are a helpful assistant..."
    )
""")

# Model with system instruction
tutor_model = genai.GenerativeModel(
    model_name          = "gemini-1.5-flash",
    system_instruction  = """You are Alex, an expert GenAI engineering tutor.
You are helping Prateek Kumar, a 2nd year B.Tech AIML student.
Rules:
- Always explain in simple terms first
- Use Python code examples when helpful
- Be encouraging and motivating
- Keep responses concise and practical
- Always end with one follow up question to check understanding"""
)

tutor_chat = tutor_model.start_chat(history=[])

questions = [
    "What is a vector database and why do we need it?",
    "How is RAG different from fine tuning an LLM?",
]

print("Tutor model with system instruction:")
print("=" * 60)
for q in questions:
    print(f"\nPrateek: {q}")
    response = tutor_chat.send_message(q)
    print(f"Alex   : {response.text}")
    print("-" * 60)


# ------------------------------------------
# PART 5 - JSON MODE
# ------------------------------------------

print("\n===== PART 5: JSON Mode =====")

print("""
JSON MODE:
    Force Gemini to output valid JSON
    Essential for production applications
    Easier to parse than extracting from text

    generation_config = genai.GenerationConfig(
        response_mime_type = "application/json"
    )
""")

json_model = genai.GenerativeModel("gemini-1.5-flash")

# Extract structured data
texts_to_analyze = [
    "Apple iPhone 15 Pro. Price: 134900 rupees. Rating: 4.5/5. Great camera, premium build quality.",
    "Samsung Galaxy S24 Ultra. Price: 129999 rupees. Rating: 4.3/5. Excellent display, S Pen included.",
    "OnePlus 12. Price: 64999 rupees. Rating: 4.4/5. Fast charging, smooth performance, great value.",
]

print("JSON extraction from product descriptions:")
for text in texts_to_analyze:
    prompt = f"""Extract product information from this text and return as JSON.

Text: {text}

Return JSON with these fields exactly:
{{
    "product_name": "",
    "price_rupees": 0,
    "rating": 0.0,
    "key_features": []
}}"""

    response = json_model.generate_content(
        prompt,
        generation_config = genai.GenerationConfig(
            response_mime_type = "application/json",
            temperature        = 0.1,
        )
    )

    data = json.loads(response.text)
    print(f"\nText: {text[:50]}...")
    print(f"Extracted JSON:")
    print(f"  Product  : {data.get('product_name')}")
    print(f"  Price    : {data.get('price_rupees')}")
    print(f"  Rating   : {data.get('rating')}")
    print(f"  Features : {data.get('key_features')}")


# ------------------------------------------
# PART 6 - FUNCTION CALLING
# ------------------------------------------

print("\n===== PART 6: Function Calling =====")

print("""
FUNCTION CALLING:
    Let Gemini call your Python functions
    Model decides when and how to call functions
    Foundation of AI agents

    1. Define function schemas
    2. Pass schemas to model
    3. Model generates function call with args
    4. You execute the actual function
    5. Return result to model
    6. Model generates final response

    This is how AI agents work under the hood!
""")

# Define functions for Gemini to call
def get_weather(city: str) -> dict:
    weather_data = {
        "Bhopal"    : {"temp": 35, "condition": "Sunny",  "humidity": 45},
        "Delhi"     : {"temp": 38, "condition": "Hot",    "humidity": 30},
        "Mumbai"    : {"temp": 32, "condition": "Humid",  "humidity": 85},
        "Bangalore" : {"temp": 25, "condition": "Cloudy", "humidity": 70},
        "Jodhpur"   : {"temp": 40, "condition": "Dry",    "humidity": 20},
    }
    return weather_data.get(
        city,
        {"temp": 30, "condition": "Unknown", "humidity": 50}
    )

def calculate_study_hours(days_remaining: int,
                           topics_left: int) -> dict:
    hours_per_topic = 4
    total_hours     = topics_left * hours_per_topic
    hours_per_day   = total_hours / max(days_remaining, 1)
    return {
        "total_hours_needed"  : total_hours,
        "hours_per_day"       : round(hours_per_day, 1),
        "is_achievable"       : hours_per_day <= 8,
        "recommendation"      : f"Study {round(hours_per_day, 1)} hours daily"
    }

def get_course_info(topic: str) -> dict:
    courses = {
        "langchain": {
            "name"      : "LangChain for LLM Application Development",
            "provider"  : "DeepLearning.AI",
            "duration"  : "1 week",
            "level"     : "Intermediate",
            "free"      : True,
        },
        "rag": {
            "name"      : "Building Systems with the ChatGPT API",
            "provider"  : "DeepLearning.AI",
            "duration"  : "1 week",
            "level"     : "Intermediate",
            "free"      : True,
        },
        "pytorch": {
            "name"      : "PyTorch for Deep Learning",
            "provider"  : "fast.ai",
            "duration"  : "2 weeks",
            "level"     : "Beginner",
            "free"      : True,
        },
    }
    return courses.get(topic.lower(), {
        "name"    : f"Course on {topic}",
        "provider": "Various platforms",
        "duration": "Varies",
        "level"   : "All levels",
        "free"    : False,
    })

# Define function schemas for Gemini
function_schemas = [
    {
        "name"       : "get_weather",
        "description": "Get current weather for a city in India",
        "parameters" : {
            "type"      : "object",
            "properties": {
                "city": {
                    "type"       : "string",
                    "description": "Name of the Indian city"
                }
            },
            "required": ["city"]
        }
    },
    {
        "name"       : "calculate_study_hours",
        "description": "Calculate study hours needed to complete learning plan",
        "parameters" : {
            "type"      : "object",
            "properties": {
                "days_remaining": {
                    "type"       : "integer",
                    "description": "Number of days left in study plan"
                },
                "topics_left": {
                    "type"       : "integer",
                    "description": "Number of topics still to be covered"
                }
            },
            "required": ["days_remaining", "topics_left"]
        }
    },
    {
        "name"       : "get_course_info",
        "description": "Get information about a learning course on a topic",
        "parameters" : {
            "type"      : "object",
            "properties": {
                "topic": {
                    "type"       : "string",
                    "description": "The topic to find courses for"
                }
            },
            "required": ["topic"]
        }
    }
]

# Function registry
function_registry = {
    "get_weather"           : get_weather,
    "calculate_study_hours" : calculate_study_hours,
    "get_course_info"       : get_course_info,
}

def call_gemini_with_functions(user_message, functions,
                                function_registry):
    tools  = [{"function_declarations": functions}]
    model  = genai.GenerativeModel(
        "gemini-1.5-flash", tools=tools)
    chat   = model.start_chat()

    print(f"User: {user_message}")

    # First call - model may want to call a function
    response = chat.send_message(user_message)

    # Check if model wants to call a function
    while response.candidates[0].content.parts:
        part = response.candidates[0].content.parts[0]

        if hasattr(part, "function_call") and part.function_call.name:
            func_name = part.function_call.name
            func_args = dict(part.function_call.args)

            print(f"Gemini calling function: {func_name}({func_args})")

            # Execute the actual function
            if func_name in function_registry:
                func_result = function_registry[func_name](**func_args)
                print(f"Function result: {func_result}")

                # Send result back to Gemini
                response = chat.send_message(
                    genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name   = func_name,
                            response={"result": func_result},
                        )
                    )
                )
            else:
                break
        else:
            break

    return response.text

# Test function calling
test_queries = [
    "What is the weather like in Bhopal right now?",
    "I have 21 days left and 6 topics to cover. How many hours should I study daily?",
    "What is a good free course for learning LangChain?",
]

print("Function Calling Examples:")
print("=" * 60)
for query in test_queries:
    print(f"\nQuery: {query}")
    result = call_gemini_with_functions(
        query, function_schemas, function_registry)
    print(f"Final response: {result}")
    print("-" * 60)


# ------------------------------------------
# PART 7 - STREAMING
# ------------------------------------------

print("\n===== PART 7: Streaming Responses =====")

print("""
STREAMING:
    Get response tokens as they are generated
    Better user experience for long responses
    User sees output immediately instead of waiting

    response = model.generate_content(
        prompt, stream=True)
    for chunk in response:
        print(chunk.text, end="", flush=True)
""")

stream_model = genai.GenerativeModel("gemini-1.5-flash")

prompt = "Explain the complete architecture of a RAG system step by step."

print(f"Streaming response to: '{prompt}'")
print("-" * 60)

response = stream_model.generate_content(
    prompt, stream=True)

full_response = ""
for chunk in response:
    if chunk.text:
        print(chunk.text, end="", flush=True)
        full_response += chunk.text

print(f"\n\nStreaming complete!")
print(f"Total characters: {len(full_response)}")


# ------------------------------------------
# PART 8 - GEMINI WITH LANGCHAIN ADVANCED
# ------------------------------------------

print("\n===== PART 8: Gemini with LangChain Advanced =====")

# LangChain Gemini setup
lc_gemini = ChatGoogleGenerativeAI(
    model          = "gemini-1.5-flash",
    temperature    = 0.7,
    google_api_key = GOOGLE_API_KEY,
)

# Build a multi step analysis chain
analysis_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are an expert technical writer."),
        ("human", "Analyze this code and provide: "
                  "1. What it does 2. Potential bugs "
                  "3. Improvements\n\nCode:\n{code}")
    ])
    | lc_gemini
    | StrOutputParser()
)

sample_code = """
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total = total + num
    return total / 10

result = calculate_average([10, 20, 30, 40, 50])
print(result)
"""

print("Code Analysis Chain:")
print(f"Code:\n{sample_code}")
analysis = analysis_chain.invoke({"code": sample_code})
print(f"\nAnalysis:\n{analysis}")


# ------------------------------------------
# MINI PROJECT - Smart Research Assistant
# ------------------------------------------

print("\n===== MINI PROJECT: Smart Research Assistant =====")

class SmartResearchAssistant:
    def __init__(self, api_key):
        self.model = genai.GenerativeModel(
            model_name          = "gemini-1.5-flash",
            system_instruction  = """You are a smart research assistant
specialized in AI and machine learning.
You help researchers and students understand complex topics.
Always provide accurate, well structured responses.
When asked to compare, use tables.
When explaining concepts, use bullet points.
Always cite whether information might need verification."""
        )
        self.chat          = self.model.start_chat(history=[])
        self.research_log  = []

    def research(self, query):
        response = self.chat.send_message(query)
        self.research_log.append({
            "query"   : query,
            "response": response.text,
        })
        return response.text

    def summarize_topic(self, topic):
        prompt = f"""Create a structured summary of {topic} with:
1. Definition (2 sentences)
2. Key concepts (bullet points)
3. Real world applications (3 examples)
4. Resources to learn more"""
        return self.research(prompt)

    def compare_concepts(self, concept1, concept2):
        prompt = f"""Compare {concept1} and {concept2}.
Create a comparison table with these rows:
- Purpose
- How it works
- Advantages
- Disadvantages
- Best used for"""
        return self.research(prompt)

    def generate_learning_path(self, goal, timeframe):
        prompt = f"""Create a detailed learning path for:
Goal: {goal}
Timeframe: {timeframe}

Include:
- Week by week breakdown
- Key resources (free ones preferred)
- Projects to build
- Skills you will gain"""
        return self.research(prompt)

    def get_research_stats(self):
        return {
            "queries_made"      : len(self.research_log),
            "topics_researched" : [
                log["query"][:30] for log in self.research_log
            ]
        }

assistant = SmartResearchAssistant(GOOGLE_API_KEY)

print("Smart Research Assistant Demo:")
print("=" * 60)

# Research RAG
print("\n1. Researching RAG Systems...")
summary = assistant.summarize_topic("RAG (Retrieval Augmented Generation)")
print(summary)

# Compare BERT vs GPT
print("\n2. Comparing BERT vs GPT...")
comparison = assistant.compare_concepts("BERT", "GPT")
print(comparison)

# Learning path
print("\n3. Generating learning path...")
path = assistant.generate_learning_path(
    goal      = "Become a GenAI Engineer",
    timeframe = "3 months"
)
print(path)

stats = assistant.get_research_stats()
print(f"\nResearch Session Stats:")
print(f"  Total queries    : {stats['queries_made']}")
print(f"  Topics researched: {len(stats['topics_researched'])}")


print("\n===== WHAT I LEARNED TODAY =====")
print("Gemini model family and free tier limits")
print("Direct Gemini API calls with generation config")
print("Multi turn chat sessions with history")
print("System instructions for model behavior")
print("JSON mode for structured output")
print("Function calling for AI agent foundation")
print("Streaming for better user experience")
print("Advanced LangChain chains with Gemini")
print("Mini Project - Smart Research Assistant")
print("\nDay 37 Done! Tomorrow - Vector Databases!")