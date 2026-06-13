
import os
import sys
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from config import GOOGLE_API_KEY, LLM_MODEL


ANSWER_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful study assistant. Use the context below to answer the question.
If the answer is not in the context, say "I couldn't find that in the document."

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Answer:
""")

CONDENSE_PROMPT = ChatPromptTemplate.from_template("""
Given the chat history and a follow-up question, rewrite the question
to be standalone (no pronouns that need history to understand).
If the question is already standalone, return it as-is.

Chat History:
{chat_history}

Follow-up question: {question}

Standalone question:
""")

SUMMARY_PROMPT = ChatPromptTemplate.from_template("""
Summarize the main topics in this document content in 5 clear bullet points.
Be concise and informative.

Content:
{context}

Summary:
""")


class RAGEngine:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model          = LLM_MODEL,
            google_api_key = GOOGLE_API_KEY,
            temperature    = 0.3,
        )
        self.chat_history: List[Dict] = []
        print(f"✅ RAG Engine ready. LLM: {LLM_MODEL}")

    def _format_history(self) -> str:
        if not self.chat_history:
            return "None"
        lines = []
        for turn in self.chat_history[-6:]:
            lines.append(f"Human: {turn['question']}")
            lines.append(f"Assistant: {turn['answer']}")
        return "\n".join(lines)

    def condense_question(self, question: str) -> str:
        if not self.chat_history:
            return question
        try:
            chain = CONDENSE_PROMPT | self.llm | StrOutputParser()
            return chain.invoke({
                "chat_history": self._format_history(),
                "question"    : question,
            })
        except Exception:
            return question

    def answer(self, question: str, docs: List[Document]) -> Dict:
        if not docs:
            return {
                "answer" : "I couldn't find relevant content in the document.",
                "sources": [],
                "chunks" : 0,
            }

        context = "\n\n---\n\n".join(doc.page_content for doc in docs)
        sources = list({doc.metadata.get("file_name", "Unknown") for doc in docs})

        try:
            chain  = ANSWER_PROMPT | self.llm | StrOutputParser()
            answer = chain.invoke({
                "context"     : context,
                "chat_history": self._format_history(),
                "question"    : question,
            })
            self.chat_history.append({"question": question, "answer": answer})
            return {"answer": answer, "sources": sources, "chunks": len(docs)}
        except Exception as e:
            return {"answer": f"Error generating answer: {e}", "sources": [], "chunks": 0}

    def summarise(self, docs: List[Document]) -> str:
        context = "\n\n".join(doc.page_content for doc in docs)
        try:
            chain = SUMMARY_PROMPT | self.llm | StrOutputParser()
            return chain.invoke({"context": context})
        except Exception as e:
            return f"Error generating summary: {e}"

    def clear_history(self):
        self.chat_history = []

    def get_stats(self) -> Dict:
        return {"history_turns": len(self.chat_history)}