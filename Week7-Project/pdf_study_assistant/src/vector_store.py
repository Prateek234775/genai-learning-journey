
import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from config import EMBEDDING_MODEL, FAISS_INDEX_PATH


class VectorStore:
    def __init__(self):
        print(f"🔄 Loading embedding model: {EMBEDDING_MODEL} ...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name    = EMBEDDING_MODEL,
            model_kwargs  = {"device": "cpu"},
            encode_kwargs = {"normalize_embeddings": True},
        )
        self.index: Optional[FAISS] = None
        print("✅ Embedding model ready.")

    def add_documents(self, documents: List[Document]) -> bool:
        try:
            if self.index is None:
                self.index = FAISS.from_documents(documents, self.embeddings)
            else:
                self.index.add_documents(documents)
            print(f"✅ Added {len(documents)} chunks to vector store.")
            return True
        except Exception as e:
            print(f"❌ Error adding documents: {e}")
            return False

    def search(self, query: str, top_k: int = 5) -> List[Document]:
        if self.index is None:
            return []
        try:
            return self.index.similarity_search(query, k=top_k)
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []

    def save(self, path: str = FAISS_INDEX_PATH):
        if self.index:
            os.makedirs(path, exist_ok=True)
            self.index.save_local(path)
            print(f"💾 Index saved to {path}")

    def load(self, path: str = FAISS_INDEX_PATH) -> bool:
        if not os.path.exists(path):
            return False
        try:
            self.index = FAISS.load_local(
                path, self.embeddings, allow_dangerous_deserialization=True)
            print(f"✅ Loaded index from {path}")
            return True
        except Exception as e:
            print(f"❌ Error loading index: {e}")
            return False

    def clear(self):
        self.index = None
        print("🗑️ Vector store cleared.")

    def get_stats(self) -> dict:
        return {
            "has_index" : self.index is not None,
            "doc_count" : self.index.index.ntotal if self.index else 0,
        }
