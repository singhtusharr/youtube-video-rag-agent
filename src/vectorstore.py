"""Step 2: embed chunks and store them in FAISS for similarity search.

Embeddings stay on HuggingFace (local, free, no API cost) — only the LLM calls
go through OpenAI. FAISS is an in-memory vector index: fast approximate nearest-
neighbor search over the embedding vectors, no external DB needed for a project
this size.
"""
from functools import lru_cache

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Loaded once and reused — the model weights are ~90MB, don't reload per request."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_vector_store(chunks: list[Document]) -> FAISS:
    return FAISS.from_documents(chunks, get_embeddings())
