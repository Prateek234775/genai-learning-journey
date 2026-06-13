
import os
import sys
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import CHUNK_SIZE, CHUNK_OVERLAP


class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size    = CHUNK_SIZE,
            chunk_overlap = CHUNK_OVERLAP,
            separators    = ["\n\n", "\n", ".", " ", ""],
        )
        self.loaded_files: List[Dict] = []

    def load_pdf(self, file_path: str) -> List[Document]:
        try:
            loader = PyPDFLoader(file_path)
            docs   = loader.load()
            for doc in docs:
                doc.metadata["file_type"] = "pdf"
                doc.metadata["file_name"] = Path(file_path).name
            print(f"✅ Loaded PDF: {Path(file_path).name} ({len(docs)} pages)")
            return docs
        except Exception as e:
            print(f"❌ Error loading PDF: {e}")
            return []

    def load_text(self, file_path: str) -> List[Document]:
        try:
            loader = TextLoader(file_path, encoding="utf-8")
            docs   = loader.load()
            for doc in docs:
                doc.metadata["file_type"] = "txt"
                doc.metadata["file_name"] = Path(file_path).name
            print(f"✅ Loaded text: {Path(file_path).name} ({len(docs[0].page_content)} chars)")
            return docs
        except Exception as e:
            print(f"❌ Error loading text: {e}")
            return []

    def load_from_bytes(self, file_bytes: bytes, filename: str) -> List[Document]:
        os.makedirs("data", exist_ok=True)
        temp_path = f"data/temp_{filename}"

        with open(temp_path, "wb") as f:
            f.write(file_bytes)

        if filename.lower().endswith(".pdf"):
            docs = self.load_pdf(temp_path)
        else:
            docs = self.load_text(temp_path)

        if os.path.exists(temp_path):
            os.remove(temp_path)

        return docs

    def split_documents(self, documents: List[Document]) -> List[Document]:
        chunks = self.text_splitter.split_documents(documents)
        print(f"📄 Split into {len(chunks)} chunks")
        return chunks

    def process_file(self, file_path: str) -> List[Document]:
        file_path = str(file_path)

        if file_path.lower().endswith(".pdf"):
            docs = self.load_pdf(file_path)
        elif file_path.lower().endswith(".txt"):
            docs = self.load_text(file_path)
        else:
            print(f"⚠️ Unsupported file type: {file_path}")
            return []

        if not docs:
            return []

        chunks = self.split_documents(docs)
        self.loaded_files.append({
            "name"  : Path(file_path).name,
            "chunks": len(chunks),
            "pages" : len(docs),
        })
        return chunks

    def process_multiple(self, file_paths: List[str]) -> List[Document]:
        all_chunks = []
        for path in file_paths:
            chunks = self.process_file(path)
            all_chunks.extend(chunks)
        print(f"📚 Total: {len(all_chunks)} chunks from {len(file_paths)} files")
        return all_chunks

    def get_stats(self) -> Dict:
        return {
            "files_loaded" : len(self.loaded_files),
            "files"        : self.loaded_files,
            "total_chunks" : sum(f["chunks"] for f in self.loaded_files),
        }
