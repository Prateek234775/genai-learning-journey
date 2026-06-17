# ============================================
# Chatbot - PDF Study Assistant
# Author: Prateek Kumar Kuntal
# Date: 16 June 2026
# ============================================

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import List, Dict
from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_engine import RAGEngine
from config import TOP_K_RETRIEVAL, GROQ_API_KEY, LLM_MODEL


class PDFStudyAssistant:

    def __init__(self):
        self.processor = DocumentProcessor()
        self.vs        = VectorStore()
        self.rag       = RAGEngine()
        self.is_ready  = False

        print("PDF Study Assistant initialized!")
        print(f"  LLM             : Groq ({LLM_MODEL})")
        print(f"  Embedding model : all-MiniLM-L6-v2")
        print(f"  Chunk size      : 500")
        print(f"  Top K retrieval : {TOP_K_RETRIEVAL}")

    # ADD these imports at the top of chatbot.py
from features import (
    generate_quiz,
    generate_flashcards,
    export_chat_history,
    calculate_reading_time,
    get_key_terms,
)

# ADD these methods inside the PDFStudyAssistant class
# (after get_stats method)

    def create_quiz(self, n_questions: int = 3,
                    difficulty: str = "medium") -> str:
        if not self.is_ready:
            return "Please upload a document first."

        docs    = self.vs.search(
            "important concepts facts definitions", top_k=6)
        context = "\n\n".join(d.page_content for d in docs)

        return generate_quiz(
            self.rag.llm, context, n_questions, difficulty)

    def create_flashcards(self, n_cards: int = 5) -> list:
        if not self.is_ready:
            return []

        docs    = self.vs.search(
            "key terms definitions concepts", top_k=6)
        context = "\n\n".join(d.page_content for d in docs)

        return generate_flashcards(
            self.rag.llm, context, n_cards)

    def get_key_terms(self, n_terms: int = 8) -> list:
        if not self.is_ready:
            return []

        docs    = self.vs.search(
            "important terms concepts vocabulary", top_k=6)
        context = "\n\n".join(d.page_content for d in docs)

        return get_key_terms(self.rag.llm, context, n_terms)

    def get_loaded_sources(self) -> list:
        return self.vs.get_all_sources()

    def remove_document(self, source_name: str) -> bool:
        return self.vs.delete_source(source_name)

    def chat_filtered(self, question: str, source: str,
                      top_k: int = TOP_K_RETRIEVAL) -> Dict:
        """Chat with answer restricted to one specific document."""
        if not self.is_ready:
            return {
                "answer" : "Please upload a document first.",
                "sources": [],
                "chunks" : 0,
            }

        standalone = self.rag.condense_question(question)
        docs       = self.vs.search_filtered(
            standalone, source, top_k=top_k)
        result     = self.rag.answer(question, docs)

        return result

    def export_history(self, format: str = "markdown") -> str:
        messages = [
            {"role": "user", "content": h["question"]}
            for h in self.rag.history
        ]
        # Interleave with answers
        full_history = []
        for h in self.rag.history:
            full_history.append(
                {"role": "user", "content": h["question"]})
            full_history.append({
                "role"   : "assistant",
                "content": h["answer"],
                "sources": h.get("sources", []),
            })
        return export_chat_history(full_history, format)

    def load_file(self, file_path: str) -> Dict:
        print(f"\nLoading: {file_path}")

        chunks = self.processor.process_file(file_path)

        if not chunks:
            return {
                "success": False,
                "message": f"Failed to load {file_path}",
                "chunks" : 0,
            }

        success = self.vs.add_documents(chunks)

        if success:
            self.is_ready = True

        return {
            "success": success,
            "message": f"Loaded {len(chunks)} chunks",
            "chunks" : len(chunks),
        }

    def load_from_bytes(self, file_bytes: bytes,
                        filename: str) -> Dict:
        chunks = self.processor.load_from_bytes(
            file_bytes, filename)

        if not chunks:
            return {
                "success": False,
                "message": f"Failed to process {filename}",
                "chunks" : 0,
            }

        chunks_split = self.processor.split_documents(chunks)
        success      = self.vs.add_documents(chunks_split)

        if success:
            self.is_ready = True

        return {
            "success": success,
            "message": (
                f"Processed {len(chunks_split)} "
                f"chunks from {filename}"
            ),
            "chunks" : len(chunks_split),
        }

    def chat(self, question: str,
             top_k: int = TOP_K_RETRIEVAL) -> Dict:
        if not self.is_ready:
            return {
                "answer" : "Please upload a document first.",
                "sources": [],
                "chunks" : 0,
            }

        standalone = self.rag.condense_question(question)
        docs       = self.vs.search(standalone, top_k=top_k)
        result     = self.rag.answer(question, docs)

        return result

    def get_document_summary(self) -> str:
        if not self.is_ready:
            return "No documents loaded."

        docs    = self.vs.search(
            "main topics overview summary", top_k=6)
        context = "\n\n".join(
            d.page_content for d in docs)

        from langchain_groq import ChatGroq
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        llm = ChatGroq(
            model   = LLM_MODEL,
            api_key = GROQ_API_KEY,
        )

        chain = (
            ChatPromptTemplate.from_template(
                "Summarize the main topics in this "
                "document in 5 bullet points:\n\n{context}"
            )
            | llm
            | StrOutputParser()
        )

        return chain.invoke({"context": context})

    def clear_chat(self):
        self.rag.clear_history()

    def get_stats(self) -> Dict:
        return {
            "is_ready"     : self.is_ready,
            "documents"    : self.processor.get_stats(),
            "vector_store" : self.vs.get_stats(),
            "rag_engine"   : self.rag.get_stats(),
        }