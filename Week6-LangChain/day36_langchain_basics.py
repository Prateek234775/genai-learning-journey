# ============================================
# DAY 36 - LangChain Basics
# Chains, Prompts, Memory
# Author: Prateek Kumar Kuntal
# Date: 9 June 2026
# ============================================

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
)
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found. "
        "Please set it in your .env file.")

print("API key loaded successfully!")


# ------------------------------------------
# PART 1 - WHAT IS LANGCHAIN
# ------------------------------------------

print("\n===== PART 1: What is LangChain =====")

print("""
LANGCHAIN:
    Framework for building applications with LLMs
    Connects LLMs with external tools and data
    Released in 2022 by Harrison Chase
    Most popular GenAI framework in the world

WHY LANGCHAIN EXISTS:
    Calling an LLM API directly is simple
    But real applications need more:
        Memory - remember conversation history
        Chains - connect multiple LLM calls
        Tools  - search web, run code, query DB
        Agents - LLM decides which tools to use
        RAG    - connect LLM to your own data

LANGCHAIN COMPONENTS:
    Models      - LLM wrappers (Gemini, OpenAI etc)
    Prompts     - prompt templates and management
    Chains      - sequences of LLM calls
    Memory      - conversation history management
    Tools       - external capabilities
    Agents      - autonomous decision making
    Retrievers  - fetch relevant documents
    Vectorstores- store and search embeddings

WHAT WE BUILD THIS WEEK:
    Day 36 - LangChain basics, chains, memory
    Day 37 - Gemini API integration
    Day 38 - Vector databases FAISS ChromaDB
    Day 39 - RAG from scratch
    Day 40 - RAG chatbot on PDF
    Day 41 - Agents and tools
""")


# ------------------------------------------
# PART 2 - SETUP GEMINI WITH LANGCHAIN
# ------------------------------------------

print("===== PART 2: Setup Gemini with LangChain =====")

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model       = "gemini-2.0-flash",
    temperature = 0.7,
    google_api_key = GOOGLE_API_KEY,
)

print(f"Model            : gemini-1.5-flash")
print(f"Temperature      : 0.7")

# Simple test call
print(f"\nTesting Gemini connection...")
response = llm.invoke("Say hello in one sentence.")
print(f"Response: {response.content}")


# ------------------------------------------
# PART 3 - PROMPT TEMPLATES
# ------------------------------------------

print("\n===== PART 3: Prompt Templates =====")

print("""
PROMPT TEMPLATES:
    Reusable prompt structures with variables
    Separate prompt logic from application code
    Easy to version and maintain prompts
    Can include few shot examples

TYPES:
    PromptTemplate     - simple string templates
    ChatPromptTemplate - for chat models (system + human)
    FewShotPromptTemplate - with examples built in
""")

# Simple PromptTemplate
simple_template = PromptTemplate(
    input_variables = ["topic", "level"],
    template        = """Explain {topic} to a {level} student.
Use simple language and one real world analogy.
Keep the explanation under 100 words."""
)

# Test template
formatted = simple_template.format(
    topic = "gradient descent",
    level = "beginner"
)
print(f"Simple PromptTemplate:")
print(f"Formatted prompt:\n{formatted}")

response = llm.invoke(formatted)
print(f"\nGemini response:\n{response.content}")

# ChatPromptTemplate
chat_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are an expert {domain} tutor who explains "
        "concepts clearly and concisely."
    ),
    HumanMessagePromptTemplate.from_template(
        "Explain {concept} in simple terms."
    )
])

# Format chat template
messages = chat_template.format_messages(
    domain  = "machine learning",
    concept = "overfitting"
)

print(f"\nChatPromptTemplate messages:")
for msg in messages:
    print(f"  [{msg.type.upper()}]: {msg.content[:60]}...")

response = llm.invoke(messages)
print(f"\nGemini response:\n{response.content}")


# ------------------------------------------
# PART 4 - CHAINS
# ------------------------------------------

print("\n===== PART 4: Chains =====")

print("""
CHAINS:
    Connect multiple components in sequence
    Output of one step becomes input of next
    Build complex pipelines from simple parts

TYPES OF CHAINS:
    LLMChain            - prompt + LLM
    SimpleSequentialChain - one input one output steps
    SequentialChain      - multiple inputs outputs
    RouterChain          - route to different chains
    TransformChain       - transform data between steps

MODERN LANGCHAIN (LCEL):
    LangChain Expression Language
    Pipe operator | connects components
    chain = prompt | llm | output_parser
    More flexible and composable
""")

# Basic LLMChain
topic_explainer = LLMChain(
    llm    = llm,
    prompt = PromptTemplate(
        input_variables = ["topic"],
        template = "Explain {topic} in exactly 2 sentences."
    ),
    verbose = False
)

topics = [
    "transformer architecture",
    "attention mechanism",
    "vector embeddings",
]

print("LLMChain - Topic Explainer:")
for topic in topics:
    response = topic_explainer.invoke({"topic": topic})
    print(f"\nTopic: {topic}")
    print(f"Explanation: {response['text']}")

# Sequential Chain - chain multiple LLM calls
from langchain.chains import SimpleSequentialChain

# Step 1 - generate a concept explanation
step1_prompt = PromptTemplate(
    input_variables = ["topic"],
    template = "Explain {topic} in 3 sentences for a beginner."
)
step1_chain = LLMChain(llm=llm, prompt=step1_prompt)

# Step 2 - generate a quiz question from the explanation
step2_prompt = PromptTemplate(
    input_variables = ["text"],
    template = """Based on this explanation:
{text}

Generate one multiple choice question with 4 options (A B C D).
Mark the correct answer at the end."""
)
step2_chain = LLMChain(llm=llm, prompt=step2_prompt)

# Combine into sequential chain
sequential_chain = SimpleSequentialChain(
    chains  = [step1_chain, step2_chain],
    verbose = False
)

print(f"\nSequential Chain - Explain then Quiz:")
result = sequential_chain.invoke(
    {"input": "gradient descent"})
print(f"Topic: gradient descent")
print(f"\nFinal output (quiz question):")
print(result["output"])

# LCEL Style - modern LangChain
from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()

lcel_chain = (
    PromptTemplate(
        input_variables = ["concept", "audience"],
        template = "Explain {concept} to {audience} in 2 sentences."
    )
    | llm
    | output_parser
)

print(f"\nLCEL Chain (modern style):")
result = lcel_chain.invoke({
    "concept" : "neural networks",
    "audience": "a 10 year old"
})
print(f"Result: {result}")


# ------------------------------------------
# PART 5 - MEMORY
# ------------------------------------------

print("\n===== PART 5: Memory =====")

print("""
MEMORY:
    LLMs are stateless by default
    Each API call has no memory of previous calls
    Memory adds conversation history to prompts

    Without memory: every message is new conversation
    With memory   : model remembers entire chat

TYPES OF MEMORY:
    ConversationBufferMemory
        Stores all messages in full
        Simple but uses lots of tokens for long chats

    ConversationBufferWindowMemory
        Stores only last K messages
        Good balance of memory and token usage

    ConversationSummaryMemory
        Summarizes old messages to save tokens
        Best for very long conversations

    ConversationKGMemory
        Extracts knowledge graph from conversation
        Remembers facts about entities mentioned

    VectorStoreRetrieverMemory
        Stores memories as vectors
        Retrieves relevant memories by similarity
""")

# Memory Type 1 - Buffer Memory
print("--- ConversationBufferMemory ---")

buffer_memory = ConversationBufferMemory(
    return_messages = True
)

conversation = ConversationChain(
    llm     = llm,
    memory  = buffer_memory,
    verbose = False
)

# Simulate multi turn conversation
turns = [
    "Hi my name is Prateek and I am learning GenAI",
    "What topic am I learning?",
    "What is the most important thing to learn in this field?",
    "Can you remember my name and what I told you?",
]

print(f"Multi-turn conversation with Buffer Memory:")
for turn in turns:
    response = conversation.predict(input=turn)
    print(f"\nHuman   : {turn}")
    print(f"AI      : {response}")

print(f"\nMemory contents ({len(buffer_memory.chat_memory.messages)} messages stored):")
for msg in buffer_memory.chat_memory.messages:
    role    = "Human" if hasattr(msg, "type") and msg.type == "human" else "AI"
    content = msg.content[:60]
    print(f"  [{role}]: {content}...")


# Memory Type 2 - Window Memory
print(f"\n--- ConversationBufferWindowMemory (k=2) ---")

window_memory = ConversationBufferWindowMemory(
    k               = 2,    # remember only last 2 exchanges
    return_messages = True
)

window_conversation = ConversationChain(
    llm     = llm,
    memory  = window_memory,
    verbose = False
)

window_turns = [
    "I love Python programming",
    "I am learning machine learning",
    "I want to become a GenAI engineer",
    "What do you remember about me?",
]

print(f"Window Memory conversation (k=2):")
for turn in window_turns:
    response = window_conversation.predict(input=turn)
    print(f"\nHuman   : {turn}")
    print(f"AI      : {response}")
    print(f"Messages in memory: "
          f"{len(window_memory.chat_memory.messages)}")


# Memory Type 3 - Summary Memory
print(f"\n--- ConversationSummaryMemory ---")

summary_memory = ConversationSummaryMemory(
    llm             = llm,
    return_messages = False
)

summary_conversation = ConversationChain(
    llm     = llm,
    memory  = summary_memory,
    verbose = False
)

summary_turns = [
    "My name is Prateek. I study at VIT Bhopal in AIML branch.",
    "I have completed 35 days of my 56 day GenAI learning journey.",
    "This week I am learning LangChain and RAG systems.",
    "What do you know about me so far?",
]

print(f"Summary Memory conversation:")
for turn in summary_turns:
    response = summary_conversation.predict(input=turn)
    print(f"\nHuman   : {turn}")
    print(f"AI      : {response}")

print(f"\nSummary memory content:")
print(summary_memory.buffer)


# ------------------------------------------
# PART 6 - OUTPUT PARSERS
# ------------------------------------------

print("\n===== PART 6: Output Parsers =====")

print("""
OUTPUT PARSERS:
    Parse LLM output into structured formats
    Convert raw text to Python objects
    Critical for production applications

TYPES:
    StrOutputParser    - plain text
    JsonOutputParser   - parse JSON output
    PydanticOutputParser - parse into Python class
    CommaSeparatedListOutputParser - return list
    StructuredOutputParser - custom structure
""")

from langchain.output_parsers import (
    CommaSeparatedListOutputParser,
    StructuredOutputParser,
    ResponseSchema,
)
from langchain_core.output_parsers import JsonOutputParser

# List output parser
list_parser   = CommaSeparatedListOutputParser()
list_template = PromptTemplate(
    input_variables  = ["topic"],
    partial_variables = {
        "format_instructions": list_parser.get_format_instructions()
    },
    template = """{format_instructions}

List exactly 5 important Python libraries for {topic}."""
)

list_chain = list_template | llm | list_parser
result     = list_chain.invoke({"topic": "machine learning"})

print(f"List Output Parser:")
print(f"Topic: machine learning libraries")
print(f"Result type: {type(result)}")
print(f"Libraries: {result}")

# Structured output parser
response_schemas = [
    ResponseSchema(
        name        = "topic",
        description = "the main topic explained"
    ),
    ResponseSchema(
        name        = "explanation",
        description = "simple explanation in 2 sentences"
    ),
    ResponseSchema(
        name        = "example",
        description = "one real world example"
    ),
    ResponseSchema(
        name        = "difficulty",
        description = "difficulty level: beginner, intermediate, or advanced"
    ),
]

structured_parser   = StructuredOutputParser.from_response_schemas(
    response_schemas)

structured_template = PromptTemplate(
    input_variables   = ["topic"],
    partial_variables = {
        "format_instructions": structured_parser.get_format_instructions()
    },
    template = """{format_instructions}

Explain the following ML concept: {topic}"""
)

structured_chain = structured_template | llm | structured_parser
result           = structured_chain.invoke(
    {"topic": "gradient descent"})

print(f"\nStructured Output Parser:")
print(f"Topic: gradient descent")
for key, value in result.items():
    print(f"  {key:<15}: {value}")


# ------------------------------------------
# PART 7 - LCEL ADVANCED
# ------------------------------------------

print("\n===== PART 7: LCEL Advanced Patterns =====")

print("""
LCEL (LangChain Expression Language):
    Modern way to build LangChain applications
    Uses pipe operator | for composition
    Supports streaming, batch, async
    More maintainable than older chain classes

PATTERNS:
    Basic chain  : prompt | llm | parser
    Parallel     : run multiple chains at once
    Branching    : different chains for different inputs
    With fallback: fallback if primary chain fails
""")

from langchain_core.runnables import RunnableParallel

# Parallel chains - run multiple at once
explain_chain = (
    PromptTemplate.from_template(
        "Explain {concept} in one sentence.")
    | llm
    | StrOutputParser()
)

example_chain = (
    PromptTemplate.from_template(
        "Give one real world example of {concept}.")
    | llm
    | StrOutputParser()
)

analogy_chain = (
    PromptTemplate.from_template(
        "Give a simple analogy for {concept}.")
    | llm
    | StrOutputParser()
)

parallel_chain = RunnableParallel(
    explanation = explain_chain,
    example     = example_chain,
    analogy     = analogy_chain,
)

print(f"Parallel Chain execution:")
result = parallel_chain.invoke(
    {"concept": "attention mechanism"})

print(f"\nConcept: attention mechanism")
for key, value in result.items():
    print(f"\n{key.upper()}:")
    print(f"  {value}")


# ------------------------------------------
# MINI PROJECT - AIML Study Assistant
# ------------------------------------------

print("\n===== MINI PROJECT: AIML Study Assistant =====")

class AIMLStudyAssistant:
    def __init__(self, llm, student_name="Student"):
        self.llm          = llm
        self.student_name = student_name
        self.memory       = ConversationBufferWindowMemory(
            k               = 5,
            return_messages = True,
            memory_key      = "chat_history"
        )
        self.session_topics = []
        self.questions_asked = 0

        # Build the chain
        system_prompt = f"""You are Alex, an expert AI and ML tutor.
You are helping {student_name} who is a 2nd year B.Tech AIML student.
They are learning GenAI engineering through a structured 56-day plan.

Rules:
- Always be encouraging and patient
- Explain concepts simply first then go deeper if asked
- Use Python code examples when relevant
- Remember what topics you have discussed in this session
- Ask if the student understood after complex explanations"""

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                system_prompt),
            MessagesPlaceholder(
                variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template(
                "{input}"),
        ])

        self.chain = LLMChain(
            llm    = self.llm,
            prompt = prompt,
            memory = self.memory,
        )

    def ask(self, question):
        self.questions_asked += 1

        # Track topics
        topic_prompt = f"What ML/AI topic does this question touch on? Answer in 3 words max: {question}"
        topic_response = self.llm.invoke(topic_prompt)
        topic = topic_response.content.strip()
        if topic not in self.session_topics:
            self.session_topics.append(topic)

        response = self.chain.invoke({"input": question})
        return response["text"]

    def quiz_me(self, topic):
        question = (
            f"Generate one multiple choice question about "
            f"{topic} with 4 options. "
            f"This is for a B.Tech AIML student."
        )
        response = self.chain.invoke({"input": question})
        return response["text"]

    def get_session_summary(self):
        summary = self.chain.invoke({
            "input": "Summarize what topics we covered in this session and give me 3 key takeaways."
        })
        return summary["text"]

    def get_stats(self):
        return {
            "questions_asked" : self.questions_asked,
            "topics_covered"  : self.session_topics,
            "messages_in_memory": len(
                self.memory.chat_memory.messages)
        }

# Create assistant for Prateek
assistant = AIMLStudyAssistant(llm, student_name="Prateek")

print(f"AIML Study Assistant initialized for Prateek")
print(f"=" * 60)

# Simulate a study session
study_questions = [
    "What is the difference between RAG and fine tuning?",
    "When should I use LangChain vs just calling the API directly?",
    "Can you explain vector databases in simple terms?",
]

for question in study_questions:
    print(f"\nPrateek: {question}")
    answer = assistant.ask(question)
    print(f"Alex   : {answer}")
    print("-" * 60)

# Quiz time
print(f"\nPrateek: Quiz me on RAG!")
quiz_question = assistant.quiz_me("RAG systems")
print(f"Alex   : {quiz_question}")

# Session summary
print(f"\n--- Session Summary ---")
summary = assistant.get_session_summary()
print(summary)

# Stats
stats = assistant.get_stats()
print(f"\n--- Session Stats ---")
print(f"Questions asked  : {stats['questions_asked']}")
print(f"Topics covered   : {stats['topics_covered']}")
print(f"Messages in memory: {stats['messages_in_memory']}")


print("\n===== WHAT I LEARNED TODAY =====")
print("What LangChain is and why it exists")
print("Setting up Gemini with LangChain")
print("PromptTemplate and ChatPromptTemplate")
print("LLMChain and SimpleSequentialChain")
print("LCEL pipe operator for modern chains")
print("ConversationBufferMemory for full history")
print("ConversationBufferWindowMemory for last K messages")
print("ConversationSummaryMemory for long conversations")
print("Output parsers for structured responses")
print("Parallel chains with RunnableParallel")
print("Mini Project - AIML Study Assistant with memory")
print("\nDay 36 Done! Tomorrow - Gemini API deep dive!")
