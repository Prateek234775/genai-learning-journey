# ============================================
# RAG Engine - PDF Study Assistant
# Author: Prateek Kumar Kuntal
# Date: 16 June 2026
# ============================================

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import List, Dict
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from config import (
    GROQ_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    MAX_HISTORY_TURNS,
    RAG_PROMPT_TEMPLATE,
    CONDENSE_PROMPT_TEMPLATE,
)


class RAGEngine:

    def __init__(self):
        self.llm = ChatGroq(
            model       = LLM_MODEL,
            temperature = LLM_TEMPERATURE,
            api_key     = GROQ_API_KEY,
        )
        self.history      = []
        self.query_count  = 0
        self.sources_used = set()

        self.rag_chain      = self._build_rag_chain()
        self.condense_chain = self._build_condense_chain()

        print(f"RAG Engine initialized with Groq!")
        print(f"  Model : {LLM_MODEL}")

    def _build_rag_chain(self):
        prompt = ChatPromptTemplate.from_template(
            RAG_PROMPT_TEMPLATE)
        return prompt | self.llm | StrOutputParser()

    def _build_condense_chain(self):
        prompt = ChatPromptTemplate.from_template(
            CONDENSE_PROMPT_TEMPLATE)
        return prompt | self.llm | StrOutputParser()

    def _format_history(self) -> str:
        if not self.history:
            return "No previous conversation."
        recent = self.history[-MAX_HISTORY_TURNS:]
        lines  = []
        for turn in recent:
            lines.append(f"Student: {turn['question']}")
            lines.append(
                f"Assistant: {turn['answer'][:150]}...")
        return "\n".join(lines)

    def _format_docs(self,
                     docs: List[Document]) -> str:
        formatted = []
        for i, doc in enumerate(docs, 1):
            source   = doc.metadata.get(
                "file_name", "Document")
            page     = doc.metadata.get("page", "")
            page_str = (f" (Page {page+1})"
                        if page != "" else "")
            formatted.append(
                f"[Source {i}: {source}{page_str}]\n"
                f"{doc.page_content}"
            )
        return "\n\n---\n\n".join(formatted)

    def _get_sources(self,
                     docs: List[Document]) -> List[str]:
        sources = set()
        for doc in docs:
            name = doc.metadata.get(
                "file_name", "Document")
            page = doc.metadata.get("page", "")
            if page != "":
                sources.add(f"{name} (p.{page+1})")
            else:
                sources.add(name)
        return list(sources)

    def condense_question(self, question: str) -> str:
        if not self.history:
            return question
        try:
            history_text = self._format_history()
            condensed    = self.condense_chain.invoke({
                "chat_history": history_text,
                "question"    : question,
            })
            return condensed.strip()
        except Exception:
            return question

    def answer(self, question: str,
               docs: List[Document]) -> Dict:
        self.query_count += 1

        if not docs:
            return {
                "answer"     : "No documents loaded. Please upload a document first.",
                "sources"    : [],
                "chunks_used": 0,
                "question"   : question,
            }

        context = self._format_docs(docs)
        history = self._format_history()
        sources = self._get_sources(docs)

        for s in sources:
            self.sources_used.add(s)

        try:
            answer = self.rag_chain.invoke({
                "context" : context,
                "history" : history,
                "question": question,
            })
        except Exception as e:
            answer = f"Error generating answer: {str(e)}"

        self.history.append({
            "question": question,
            "answer"  : answer,
            "sources" : sources,
        })

        return {
            "answer"     : answer,
            "sources"    : sources,
            "chunks_used": len(docs),
            "question"   : question,
        }

    def clear_history(self):
        self.history = []

    def get_stats(self) -> Dict:
        return {
            "total_queries"      : self.query_count,
            "conversation_turns" : len(self.history),
            "unique_sources"     : list(self.sources_used),
        }