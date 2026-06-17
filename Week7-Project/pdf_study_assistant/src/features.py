# ============================================
# Additional Features - PDF Study Assistant
# Quiz Generator, Export, Retry Logic
# Author: Prateek Kumar Kuntal
# Date: 18 June 2026
# ============================================

import os
import sys
import time
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import List, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def generate_quiz(llm, context: str, n_questions: int = 3,
                  difficulty: str = "medium") -> str:
    """Generate multiple choice quiz from document context."""
    prompt = ChatPromptTemplate.from_template("""
Generate {n_questions} multiple choice questions based ONLY
on the following content. Difficulty level: {difficulty}.

Format each question exactly like this:
Q1: [question text]
A) [option]
B) [option]
C) [option]
D) [option]
Correct Answer: [letter]

Content:
{context}

Quiz:""")

    chain = prompt | llm | StrOutputParser()

    try:
        return chain.invoke({
            "context"    : context,
            "n_questions": n_questions,
            "difficulty" : difficulty,
        })
    except Exception as e:
        return f"Could not generate quiz: {str(e)}"


def generate_flashcards(llm, context: str,
                        n_cards: int = 5) -> List[Dict]:
    """Generate study flashcards from document context."""
    prompt = ChatPromptTemplate.from_template("""
Create {n_cards} flashcards based ONLY on the following content.
Return ONLY valid JSON, no other text, no markdown formatting.

Format:
[
  {{"front": "question or term", "back": "answer or definition"}},
  {{"front": "question or term", "back": "answer or definition"}}
]

Content:
{context}

JSON:""")

    chain = prompt | llm | StrOutputParser()

    try:
        result = chain.invoke({
            "context": context,
            "n_cards": n_cards,
        })
        # Clean up potential markdown formatting
        result = result.strip()
        if result.startswith("```"):
            result = result.split("```")[1]
            if result.startswith("json"):
                result = result[4:]
        return json.loads(result.strip())
    except Exception as e:
        print(f"Flashcard generation error: {e}")
        return []


def export_chat_history(messages: List[Dict],
                        format: str = "markdown") -> str:
    """Export conversation history to markdown or text."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    if format == "markdown":
        lines = [
            f"# PDF Study Assistant - Chat Export",
            f"Generated: {timestamp}",
            "",
            "---",
            "",
        ]
        for msg in messages:
            role = "**You**" if msg["role"] == "user" else "**Assistant**"
            lines.append(f"{role}: {msg['content']}")
            if msg.get("sources"):
                lines.append(f"*Sources: {', '.join(msg['sources'])}*")
            lines.append("")
        return "\n".join(lines)
    else:
        lines = [f"Chat Export - {timestamp}", "=" * 40, ""]
        for msg in messages:
            role = "You" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
            lines.append("")
        return "\n".join(lines)


def retry_with_backoff(func, max_retries: int = 3,
                       initial_delay: float = 1.0):
    """Decorator-like helper to retry API calls with exponential backoff."""
    def wrapper(*args, **kwargs):
        delay = initial_delay
        last_error = None

        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                error_str  = str(e).lower()

                if "rate" in error_str or "quota" in error_str:
                    print(f"Rate limit hit, retrying in "
                          f"{delay}s... (attempt {attempt+1})")
                    time.sleep(delay)
                    delay *= 2  # exponential backoff
                else:
                    raise e

        raise last_error

    return wrapper


def calculate_reading_time(text: str,
                           words_per_minute: int = 200) -> Dict:
    """Estimate reading time for document content."""
    word_count = len(text.split())
    minutes    = word_count / words_per_minute
    return {
        "word_count"  : word_count,
        "minutes"     : round(minutes, 1),
        "display"     : (f"{round(minutes)} min read"
                         if minutes >= 1
                         else "Less than 1 min read"),
    }


def get_key_terms(llm, context: str,
                  n_terms: int = 8) -> List[str]:
    """Extract key terms and concepts from document."""
    prompt = ChatPromptTemplate.from_template("""
Extract the {n_terms} most important technical terms or
concepts from this content. Return ONLY a comma separated
list of terms, nothing else.

Content:
{context}

Key terms:""")

    chain = prompt | llm | StrOutputParser()

    try:
        result = chain.invoke({
            "context": context,
            "n_terms": n_terms,
        })
        terms = [t.strip() for t in result.split(",")]
        return [t for t in terms if t][:n_terms]
    except Exception:
        return []