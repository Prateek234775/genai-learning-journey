# ============================================
# Vector Store - PDF Study Assistant
# Author: Prateek Kumar Kuntal
# Date: 16 June 2026
# ============================================

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import List
from config import (
    EMBEDDING_MODEL,
    VECTORSTORE_PATH,
    TOP_K_RETRIEVAL,
)


class VectorStore:
    def __init__(self):
        from langchain_community.embeddings import HuggingFaceEmbeddings
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embeddings  = HuggingFaceEmbeddings(
            model_name    = EMBEDDING_MODEL,
            model_kwargs  = {"device": "cpu"},
            encode_kwargs = {"normalize_embeddings": True},
        )
        self.vectorstore = None
        self.retriever   = None
        self.doc_count   = 0

    def build_from_documents(self, documents: List) -> bool:
        try:
            from langchain_community.vectorstores import FAISS
            print(f"Building vector store from "
                  f"{len(documents)} chunks...")
            self.vectorstore = FAISS.from_documents(
                documents = documents,
                embedding = self.embeddings,
            )
            self.retriever   = self.vectorstore.as_retriever(
                search_type   = "similarity",
                search_kwargs = {"k": TOP_K_RETRIEVAL},
            )
            self.doc_count   = len(documents)
            print(f"Vector store built with "
                  f"{self.doc_count} vectors!")
            return True
        except Exception as e:
            print(f"Error building vector store: {e}")
            return False

    def add_documents(self, documents: List) -> bool:
        try:
            if self.vectorstore is None:
                return self.build_from_documents(documents)
            self.vectorstore.add_documents(documents)
            self.doc_count += len(documents)
            self.retriever  = self.vectorstore.as_retriever(
                search_kwargs={"k": TOP_K_RETRIEVAL})
            print(f"Added {len(documents)} chunks. "
                  f"Total: {self.doc_count}")
            return True
        except Exception as e:
            print(f"Error adding documents: {e}")
            return False

    def search(self, query: str,
               top_k: int = TOP_K_RETRIEVAL) -> List:
        if self.vectorstore is None:
            return []
        try:
            return self.vectorstore.similarity_search(
                query, k=top_k)
        except Exception as e:
            print(f"Search error: {e}")
            return []

    def search_with_scores(self, query: str,
                            top_k: int = TOP_K_RETRIEVAL):
        if self.vectorstore is None:
            return []
        try:
            return self.vectorstore.similarity_search_with_score(
                query, k=top_k)
        except Exception as e:
            print(f"Search error: {e}")
            return []

    def get_retriever(self, top_k: int = TOP_K_RETRIEVAL):
        if self.vectorstore is None:
            return None
        return self.vectorstore.as_retriever(
            search_kwargs={"k": top_k})

    def save(self, path: str = VECTORSTORE_PATH) -> bool:
        if self.vectorstore is None:
            return False
        try:
            os.makedirs(path, exist_ok=True)
            self.vectorstore.save_local(path)
            print(f"Vector store saved to {path}")
            return True
        except Exception as e:
            print(f"Error saving: {e}")
            return False

    def load(self, path: str = VECTORSTORE_PATH) -> bool:
        try:
            from langchain_community.vectorstores import FAISS
            self.vectorstore = FAISS.load_local(
                path, self.embeddings,
                allow_dangerous_deserialization=True,
            )
            self.retriever   = self.vectorstore.as_retriever(
                search_kwargs={"k": TOP_K_RETRIEVAL})
            print(f"Vector store loaded from {path}")
            return True
        except Exception as e:
            print(f"Error loading: {e}")
            return False

    def get_all_sources(self) -> list:
        if self.vectorstore is None:
            return []
        try:
            all_docs = list(
                self.vectorstore.docstore._dict.values())
            sources  = set(
                doc.metadata.get("file_name", "unknown")
                for doc in all_docs)
            return sorted(list(sources))
        except Exception:
            return []

    def search_filtered(self, query: str, source: str,
                        top_k: int = TOP_K_RETRIEVAL):
        if self.vectorstore is None:
            return []
        try:
            all_results = self.vectorstore.similarity_search(
                query, k=top_k * 3)
            filtered = [
                doc for doc in all_results
                if doc.metadata.get("file_name") == source
            ]
            return filtered[:top_k]
        except Exception as e:
            print(f"Filtered search error: {e}")
            return []

    def is_ready(self) -> bool:
        return self.vectorstore is not None

    def get_stats(self) -> dict:
        return {
            "is_ready"  : self.is_ready(),
            "doc_count" : self.doc_count,
        }