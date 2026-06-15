# ============================================
# DAY 41 - LangChain Agents and Tools
# Author: Prateek Kumar Kuntal
# Date: 14 June 2026
# ============================================

import os
import json
import time
import math
from datetime import datetime
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import (
    AgentExecutor,
    create_tool_calling_agent,
)
from langchain.tools import tool, Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model          = "gemini-2.0-flash",
    temperature    = 0.1,   # low temp for reliable tool use
    google_api_key = GOOGLE_API_KEY,
)

print("Gemini initialized for agent use!")


# ------------------------------------------
# PART 1 - WHAT ARE AGENTS
# ------------------------------------------

print("\n===== PART 1: What are Agents =====")

print("""
AI AGENTS:
    LLM that can use tools to take actions
    Decides which tools to use and when
    Can plan and execute multi step tasks
    The next frontier of AI applications

HOW AGENTS WORK:
    1. User gives a task
    2. Agent thinks about what to do (reasoning)
    3. Agent selects a tool to use (acting)
    4. Tool executes and returns result (observation)
    5. Agent processes result and thinks again
    6. Repeat until task is complete
    7. Agent gives final answer

REACT FRAMEWORK (Reason + Act):
    Thought: I need to find the current stock price
    Action: search[Tesla stock price today]
    Observation: Tesla (TSLA) is trading at 245.32
    Thought: Now I have the price I can answer
    Final Answer: Tesla stock is at 245.32 dollars

TYPES OF AGENTS:
    Tool calling agents   - use function calling API
    ReAct agents          - reason and act in text
    Plan and execute      - plan all steps first
    Self-ask with search  - break into sub-questions

COMMON AGENT TOOLS:
    Web search    - find current information
    Calculator    - do math accurately
    Wikipedia     - get factual information
    Code executor - run Python code
    File reader   - read local files
    Database      - query databases
    API caller    - call external APIs
    Email sender  - send emails
""")


# ------------------------------------------
# PART 2 - BUILDING CUSTOM TOOLS
# ------------------------------------------

print("===== PART 2: Building Custom Tools =====")

print("""
TOOLS IN LANGCHAIN:
    Functions the agent can call
    Each tool has:
        name        - unique identifier
        description - tells agent when to use it
        function    - actual implementation

    Good tool descriptions are critical
    Agent reads description to decide which tool to use
    Be specific about what the tool does and when to use it
""")

# Tool 1 - Calculator
@tool
def calculator(expression: str) -> str:
    """
    Useful for performing mathematical calculations.
    Input should be a valid Python math expression.
    Examples: '2 + 2', '15 * 8', 'math.sqrt(144)', '2 ** 10'
    Use this whenever you need to calculate numbers accurately.
    """
    try:
        allowed_names = {
            "math"  : math,
            "abs"   : abs,
            "round" : round,
            "min"   : min,
            "max"   : max,
            "sum"   : sum,
            "pow"   : pow,
        }
        result = eval(expression, {"__builtins__": {}},
                      allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Error in calculation: {str(e)}"

# Tool 2 - Current Date and Time
@tool
def get_current_datetime(query: str = "") -> str:
    """
    Returns the current date and time.
    Use this when the user asks about today's date,
    current time, day of week, or anything time related.
    Input can be empty or any time related query.
    """
    now = datetime.now()
    return (f"Current date and time: "
            f"{now.strftime('%A, %B %d, %Y at %I:%M %p')}")

# Tool 3 - Unit Converter
@tool
def unit_converter(conversion: str) -> str:
    """
    Converts between common units of measurement.
    Input format: 'value unit1 to unit2'
    Examples:
        '100 km to miles'
        '70 kg to pounds'
        '30 celsius to fahrenheit'
        '1000 meters to feet'
    Use this for unit conversion questions.
    """
    try:
        parts = conversion.lower().split()
        value = float(parts[0])
        from_unit = parts[1]
        to_unit   = parts[3]

        conversions = {
            ("km",        "miles")      : lambda x: x * 0.621371,
            ("miles",     "km")         : lambda x: x * 1.60934,
            ("kg",        "pounds")     : lambda x: x * 2.20462,
            ("pounds",    "kg")         : lambda x: x * 0.453592,
            ("celsius",   "fahrenheit") : lambda x: x * 9/5 + 32,
            ("fahrenheit","celsius")    : lambda x: (x-32) * 5/9,
            ("meters",    "feet")       : lambda x: x * 3.28084,
            ("feet",      "meters")     : lambda x: x * 0.3048,
            ("liters",    "gallons")    : lambda x: x * 0.264172,
            ("gallons",   "liters")     : lambda x: x * 3.78541,
        }

        key    = (from_unit, to_unit)
        if key in conversions:
            result = conversions[key](value)
            return (f"{value} {from_unit} = "
                    f"{result:.4f} {to_unit}")
        else:
            return f"Conversion from {from_unit} to {to_unit} not supported."
    except Exception as e:
        return f"Error: {str(e)}. Use format: 'value unit1 to unit2'"

# Tool 4 - Study Planner
@tool
def study_planner(request: str) -> str:
    """
    Creates study plans and calculates study time.
    Input should describe what to plan.
    Examples:
        'plan 5 topics in 10 days'
        'how many hours per day for 8 topics in 2 weeks'
        'study schedule for 3 subjects exam in 7 days'
    Use this for study planning and schedule questions.
    """
    try:
        words  = request.lower().split()
        topics = 0
        days   = 0
        hours  = 4  # default hours per day

        for i, word in enumerate(words):
            if word.isdigit():
                val = int(word)
                if i+1 < len(words):
                    next_word = words[i+1]
                    if "topic" in next_word or "subject" in next_word:
                        topics = val
                    elif "day" in next_word or "week" in next_word:
                        days   = val * (7 if "week" in next_word else 1)
                    elif "hour" in next_word:
                        hours  = val

        if topics == 0: topics = 5
        if days   == 0: days   = 7

        hours_per_topic = 4
        total_hours     = topics * hours_per_topic
        hours_per_day   = total_hours / days

        plan = f"""
Study Plan:
- Topics to cover: {topics}
- Days available : {days}
- Total hours needed: {total_hours}
- Hours per day: {hours_per_day:.1f}
- Daily target: {math.ceil(topics/days)} topic(s) per day
- Status: {'Achievable' if hours_per_day <= 8 else 'Challenging - consider reducing topics'}
"""
        return plan.strip()
    except Exception as e:
        return f"Could not create plan: {str(e)}"

# Tool 5 - AIML Knowledge
@tool
def aiml_knowledge(question: str) -> str:
    """
    Answers questions about AI and Machine Learning concepts.
    Use this for questions about specific ML/AI terms,
    algorithms, model architectures, or technical concepts.
    Input should be a specific ML/AI concept or question.
    Examples:
        'What is gradient descent?'
        'Explain transformer architecture'
        'What is the difference between BERT and GPT?'
    """
    knowledge_base = {
        "gradient descent" : "An optimization algorithm that minimizes loss by moving in direction of steepest descent. Updates weights by subtracting learning rate times gradient.",
        "transformer"      : "Architecture using multi-head attention and feed-forward layers. Processes sequences in parallel unlike RNNs. Foundation of BERT and GPT models.",
        "bert"             : "Bidirectional Encoder Representations from Transformers. Encoder-only model pretrained with masked language modeling. Good for understanding tasks.",
        "gpt"              : "Generative Pretrained Transformer. Decoder-only model trained for next token prediction. Good for text generation tasks.",
        "rag"              : "Retrieval Augmented Generation. Combines vector search with LLM generation. Allows LLMs to answer from custom documents without fine tuning.",
        "lora"             : "Low Rank Adaptation. PEFT method that adds small trainable matrices to frozen model. Reduces trainable parameters by 100x or more.",
        "attention"        : "Mechanism that computes weighted sum of values using query-key similarity. Allows models to focus on relevant parts of input.",
        "overfitting"      : "Model memorizes training data and fails on new data. Solution: regularization, dropout, more data, cross-validation.",
        "langchain"        : "Framework for building LLM applications. Provides chains, agents, memory, tools, and retrievers for composing complex AI pipelines.",
        "embeddings"       : "Dense vector representations of text that capture semantic meaning. Similar texts have similar vectors. Used in RAG and semantic search.",
    }

    question_lower = question.lower()
    for key, value in knowledge_base.items():
        if key in question_lower:
            return f"{key.upper()}: {value}"

    return (f"Specific information about '{question}' "
            f"not in local knowledge base. "
            f"Try web search for more details.")

# Tool 6 - Code Analyzer
@tool
def code_analyzer(code: str) -> str:
    """
    Analyzes Python code for basic issues and provides feedback.
    Input should be Python code as a string.
    Use this when the user shares code and wants analysis,
    debugging help, or code review.
    """
    issues      = []
    suggestions = []

    lines = code.strip().split("\n")

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Check common issues
        if "except:" in stripped and "except Exception" not in stripped:
            issues.append(f"Line {i}: Bare except clause - too broad")

        if "== None" in stripped:
            issues.append(f"Line {i}: Use 'is None' instead of '== None'")

        if "== True" in stripped or "== False" in stripped:
            issues.append(f"Line {i}: Avoid comparing to True/False directly")

        if len(line) > 79:
            suggestions.append(f"Line {i}: Line too long ({len(line)} chars) - PEP8 recommends 79")

        if stripped.startswith("import") and "," in stripped:
            suggestions.append(f"Line {i}: Multiple imports on one line - split them")

    # Summary
    result = f"Code Analysis ({len(lines)} lines):\n"
    if issues:
        result += f"\nIssues found ({len(issues)}):\n"
        for issue in issues:
            result += f"  - {issue}\n"
    else:
        result += "\nNo major issues found.\n"

    if suggestions:
        result += f"\nSuggestions ({len(suggestions)}):\n"
        for s in suggestions:
            result += f"  - {s}\n"

    result += f"\nOverall: {'Needs fixes' if issues else 'Good code'}"
    return result

# Test tools independently
print("Testing custom tools:")
print(f"\nCalculator: {calculator.invoke('2 ** 10 + math.sqrt(144)')}")
print(f"\nDatetime  : {get_current_datetime.invoke('')}")
print(f"\nConverter : {unit_converter.invoke('100 km to miles')}")
print(f"\nPlanner   : {study_planner.invoke('6 topics in 14 days')}")
print(f"\nAI Knowledge: {aiml_knowledge.invoke('What is RAG?')}")

sample_code = """
def process(data):
    try:
        result = data / 0
    except:
        pass
    if result == None:
        return False == True
"""
print(f"\nCode analysis: {code_analyzer.invoke(sample_code)}")


# ------------------------------------------
# PART 3 - WEB SEARCH AND WIKIPEDIA TOOLS
# ------------------------------------------

print("\n===== PART 3: Web Search and Wikipedia Tools =====")

print("""
SEARCH TOOLS:
    DuckDuckGo - free web search, no API key needed
    Wikipedia  - factual information retrieval
    Google     - needs API key, most comprehensive

    These tools let agents access real-time
    information beyond their training cutoff
""")

# DuckDuckGo search
search_tool = DuckDuckGoSearchRun()
search_tool.name        = "web_search"
search_tool.description = (
    "Search the web for current information. "
    "Use for recent events, news, current prices, "
    "or any information that might have changed recently. "
    "Input should be a search query."
)

# Wikipedia tool
wikipedia = WikipediaAPIWrapper(top_k_results=2)
wiki_tool = Tool(
    name        = "wikipedia",
    func        = wikipedia.run,
    description = (
        "Search Wikipedia for factual information about "
        "people, places, concepts, history, and science. "
        "Good for established knowledge and definitions. "
        "Input should be the topic to search."
    )
)

# Test search tools
print("Testing web search...")
try:
    result = search_tool.run("LangChain latest version 2025")
    print(f"Web search result: {result[:200]}...")
except Exception as e:
    print(f"Web search error: {e}")

print("\nTesting Wikipedia...")
try:
    result = wiki_tool.run("transformer neural network")
    print(f"Wikipedia result: {result[:200]}...")
except Exception as e:
    print(f"Wikipedia error: {e}")


# ------------------------------------------
# PART 4 - CREATE AGENT
# ------------------------------------------

print("\n===== PART 4: Create Agent =====")

print("""
TOOL CALLING AGENT:
    Most reliable agent type for modern LLMs
    Uses function calling API under the hood
    Model decides which tool to call and with what args
    More structured than text-based ReAct agents

AGENT COMPONENTS:
    Tools   - list of available tools
    Prompt  - includes chat history and agent scratchpad
    LLM     - the reasoning model
    Executor- runs the agent loop
""")

# Collect all tools
all_tools = [
    calculator,
    get_current_datetime,
    unit_converter,
    study_planner,
    aiml_knowledge,
    code_analyzer,
    search_tool,
    wiki_tool,
]

print(f"Tools available to agent: {len(all_tools)}")
for t in all_tools:
    print(f"  - {t.name}: {t.description[:60]}...")

# Agent prompt
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant for students
learning AI and Machine Learning. You have access to various
tools to help answer questions accurately.

Guidelines:
- Use tools when you need current information or calculations
- Think step by step before using tools
- Use the most appropriate tool for each task
- If a tool fails try another approach
- Always provide clear and helpful answers
- For ML/AI questions check aiml_knowledge first
- For math use calculator tool not mental math
- For current events or recent info use web_search"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create agent
agent = create_tool_calling_agent(
    llm     = llm,
    tools   = all_tools,
    prompt  = agent_prompt,
)

# Create executor
agent_executor = AgentExecutor(
    agent         = agent,
    tools         = all_tools,
    verbose       = True,
    max_iterations= 5,
    return_intermediate_steps = True,
)

print("\nAgent created successfully!")


# ------------------------------------------
# PART 5 - TEST AGENT
# ------------------------------------------

print("\n===== PART 5: Test Agent =====")

def run_agent(query, executor, history=None):
    print(f"\nQuery: {query}")
    print("-" * 50)

    try:
        result = executor.invoke({
            "input"       : query,
            "chat_history": history or [],
        })

        answer = result.get("output", "No answer")
        steps  = result.get("intermediate_steps", [])

        print(f"\nFinal Answer: {answer}")

        if steps:
            print(f"\nTools used: {len(steps)}")
            for i, (action, observation) in enumerate(steps):
                print(f"  Step {i+1}: {action.tool} -> "
                      f"{str(observation)[:80]}...")

        return answer

    except Exception as e:
        print(f"Agent error: {e}")
        return f"Error: {str(e)}"

# Test various queries
test_queries = [
    "What is today's date and what day of the week is it?",
    "Calculate 15% of 85000 and also 2 to the power of 15",
    "Convert 5 miles to kilometers and 75 kg to pounds",
    "I have 8 ML topics to study in 3 weeks. Create a study plan.",
    "Explain what RAG is in machine learning.",
]

print("Testing agent with various queries:")
for query in test_queries:
    run_agent(query, agent_executor)
    print("=" * 60)
    time.sleep(2)  # avoid rate limits


# ------------------------------------------
# PART 6 - AGENT WITH MEMORY
# ------------------------------------------

print("\n===== PART 6: Agent with Memory =====")

print("""
AGENT MEMORY:
    By default agents have no memory
    Each query starts fresh conversation
    Adding memory enables multi-turn conversations
    Agent can refer to previous questions and answers
""")

from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, AIMessage

class MemoryAgent:
    def __init__(self, llm, tools):
        self.llm         = llm
        self.tools       = tools
        self.history     = []
        self.query_count = 0

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are Alex, a helpful AI tutor
for students learning machine learning and GenAI.
You remember the conversation history and use it
to provide contextual helpful answers.
Use tools when needed for accurate information."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(
            llm    = llm,
            tools  = tools,
            prompt = prompt,
        )

        self.executor = AgentExecutor(
            agent          = agent,
            tools          = tools,
            verbose        = False,
            max_iterations = 4,
        )

    def chat(self, message):
        self.query_count += 1

        result = self.executor.invoke({
            "input"       : message,
            "chat_history": self.history,
        })

        answer = result.get("output", "")

        # Update history
        self.history.append(HumanMessage(content=message))
        self.history.append(AIMessage(content=answer))

        # Keep last 10 messages
        if len(self.history) > 10:
            self.history = self.history[-10:]

        return answer

    def get_stats(self):
        return {
            "queries"        : self.query_count,
            "history_length" : len(self.history),
        }

# Create memory agent
memory_agent = MemoryAgent(llm, all_tools)

# Multi turn conversation
conversation = [
    "Hi I am Prateek. I am learning about RAG systems.",
    "Can you explain what a vector database is?",
    "What topic am I currently studying?",
    "How many hours should I study if I have 5 more topics and 10 days left?",
    "What is my name again?",
]

print("Memory Agent Conversation:")
print("=" * 70)
for message in conversation:
    print(f"\nPrateek: {message}")
    response = memory_agent.chat(message)
    print(f"Alex   : {response}")
    print("-" * 70)
    time.sleep(2)

stats = memory_agent.get_stats()
print(f"\nConversation stats: {stats}")


# ------------------------------------------
# PART 7 - MULTI TOOL AGENT TASK
# ------------------------------------------

print("\n===== PART 7: Complex Multi-Tool Task =====")

print("""
COMPLEX TASKS:
    Agents can chain multiple tool calls
    Each tool result informs next decision
    This is where agents truly shine

    Example complex task:
    "Research transformers, find a Wikipedia summary,
     calculate how many attention heads in GPT-3 given
     d_model=12288 and d_k=128, then create a study plan"

    Agent will:
    1. Use wikipedia tool to research transformers
    2. Use calculator for the math
    3. Use study_planner for the schedule
    4. Combine all results into final answer
""")

complex_task = """
I am preparing for an ML interview next week.
Can you:
1. Tell me today's date and how many days until next Monday
2. Give me a quick summary of what transformer architecture is
3. Calculate: if I study 3 hours per day for 6 days how many total hours is that
4. Create a study plan for 4 topics in 6 days
Please combine all this information into a helpful interview prep response.
"""

print(f"Complex multi-tool task:")
print(f"Task: {complex_task[:100]}...")
print(f"\nAgent executing...")

result = run_agent(complex_task, agent_executor)


# ------------------------------------------
# MINI PROJECT - Personal AI Study Agent
# ------------------------------------------

print("\n===== MINI PROJECT: Personal AI Study Agent =====")

# Additional specialized tools
@tool
def get_topic_resources(topic: str) -> str:
    """
    Gets learning resources for a specific ML/AI topic.
    Input should be the topic name.
    Examples: 'transformers', 'RAG', 'fine tuning', 'LangChain'
    Use when user asks for learning resources or how to learn a topic.
    """
    resources = {
        "transformers"  : "1. Illustrated Transformer by Jay Alammar\n2. Attention is All You Need paper\n3. HuggingFace Transformers course (free)",
        "rag"           : "1. LangChain RAG tutorial docs\n2. DeepLearning.AI LangChain course (free)\n3. Jerry Liu LlamaIndex tutorials",
        "fine tuning"   : "1. HuggingFace PEFT documentation\n2. LoRA paper by Hu et al\n3. DeepLearning.AI fine tuning course",
        "langchain"     : "1. LangChain official docs\n2. LangChain YouTube channel\n3. Udemy LangChain bootcamp",
        "pytorch"       : "1. PyTorch official tutorials\n2. fast.ai practical deep learning\n3. Andrej Karpathy YouTube",
        "machine learning": "1. Andrew Ng ML Specialization Coursera\n2. fast.ai ML course\n3. Hands-On ML by Aurelien Geron",
    }

    topic_lower = topic.lower()
    for key, value in resources.items():
        if key in topic_lower or topic_lower in key:
            return f"Resources for {topic}:\n{value}"

    return (f"For {topic} try: Coursera, fast.ai, "
            f"YouTube (Krish Naik, Andrej Karpathy), "
            f"HuggingFace docs, Papers with Code")

@tool
def progress_tracker(update: str) -> str:
    """
    Tracks study progress and provides encouragement.
    Input should describe what was completed today.
    Example: 'completed RAG tutorial and built chatbot'
    Use when user wants to log their progress.
    """
    now   = datetime.now().strftime("%B %d, %Y")
    entry = f"[{now}] {update}"

    # Load existing progress
    progress_file = "study_progress.json"
    progress      = []

    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            progress = json.load(f)

    progress.append(entry)

    # Save progress
    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)

    return (f"Progress logged! Entry #{len(progress)}\n"
            f"'{entry}'\n"
            f"Keep going! You have logged {len(progress)} "
            f"study sessions total.")

# Extended tool list for personal agent
personal_tools = all_tools + [
    get_topic_resources,
    progress_tracker,
]

# Personal agent system prompt
personal_system = f"""You are Alex, Prateek's personal AI study coach.
Prateek is a 2nd year B.Tech AIML student on a 56-day GenAI learning journey.
Today is day 41. He has completed Python, ML, Deep Learning, NLP,
HuggingFace, and LangChain basics.

Your role:
- Help Prateek study effectively
- Answer ML/AI questions using your knowledge tool
- Calculate study schedules when asked
- Find resources for topics he wants to learn
- Track his progress when he tells you what he completed
- Be encouraging and motivating

Always be specific, practical, and encouraging."""

personal_prompt = ChatPromptTemplate.from_messages([
    ("system", personal_system),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

personal_agent = create_tool_calling_agent(
    llm    = llm,
    tools  = personal_tools,
    prompt = personal_prompt,
)

personal_executor = AgentExecutor(
    agent          = personal_agent,
    tools          = personal_tools,
    verbose        = False,
    max_iterations = 5,
)

personal_history = []

def chat_with_agent(message, executor, history):
    result = executor.invoke({
        "input"       : message,
        "chat_history": history,
    })
    answer = result.get("output", "")
    history.append(HumanMessage(content=message))
    history.append(AIMessage(content=answer))
    return answer

print("Personal AI Study Agent for Prateek:")
print("=" * 70)

personal_queries = [
    "What topics should I focus on in my remaining 15 days?",
    "I just completed building a PDF RAG chatbot today. Log my progress.",
    "What resources should I use to learn about LangChain agents?",
    "I have 4 topics left and 14 days. How should I plan my studies?",
    "What did I complete today?",
]

for query in personal_queries:
    print(f"\nPrateek: {query}")
    response = chat_with_agent(
        query, personal_executor, personal_history)
    print(f"Alex   : {response}")
    print("-" * 70)
    time.sleep(2)


print("\n===== WHAT I LEARNED TODAY =====")
print("What AI agents are and how they work")
print("ReAct framework for agent reasoning")
print("Building custom tools with @tool decorator")
print("Web search and Wikipedia tools")
print("Creating tool calling agents with LangChain")
print("Agent executor with verbose output")
print("Adding memory to agents for multi-turn chat")
print("Complex multi-tool task execution")
print("Building personal AI study coach agent")
print("\nDay 41 Done! Tomorrow is REST DAY!")